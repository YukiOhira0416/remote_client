#include "AudioReceiver.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iomanip>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "DebugLog.h"
#include "Globals.h"
#include "ReedSolomon.h"

using namespace DebugLogAsync;

namespace {
#pragma pack(push, 1)
struct AudioPacketHeader {
    char magic[4];        // "AUD0"
    uint8_t version;      // 1
    uint8_t endian;       // 1=little, 2=big
    uint16_t headerSize;  // 52
    uint16_t flags;
    uint32_t streamId;
    uint32_t frameId;
    uint64_t captureTimeNs;
    uint32_t sampleRate;
    uint16_t channels;
    uint16_t bitsPerSample;
    uint16_t frameDurationUs;
    uint16_t shardIndex;
    uint8_t fecK;
    uint8_t fecM;
    uint16_t reserved;
    uint16_t shardSize;
    uint32_t originalBytes;
    uint32_t crc32;
};
#pragma pack(pop)

static_assert(sizeof(AudioPacketHeader) == 52, "AudioPacketHeader size mismatch");

// CRC32 implementation (polynomial 0xEDB88320) shared by all calculations here.
uint32_t ComputeCrc32(const uint8_t* data, size_t length) {
    static uint32_t table[256];
    static bool tableInit = false;
    if (!tableInit) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int j = 0; j < 8; ++j) {
                c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            table[i] = c;
        }
        tableInit = true;
    }

    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < length; ++i) {
        crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

uint16_t ReadU16(const uint8_t* ptr, uint8_t endian) {
    uint16_t v;
    memcpy(&v, ptr, sizeof(uint16_t));
    if (endian == 2) {
        v = ntohs(v);
    }
    return v;
}

uint32_t ReadU32(const uint8_t* ptr, uint8_t endian) {
    uint32_t v;
    memcpy(&v, ptr, sizeof(uint32_t));
    if (endian == 2) {
        v = ntohl(v);
    }
    return v;
}

uint64_t ReadU64(const uint8_t* ptr, uint8_t endian) {
    uint64_t v;
    memcpy(&v, ptr, sizeof(uint64_t));
    if (endian == 2) {
        uint32_t hi, lo;
        memcpy(&hi, ptr, sizeof(uint32_t));
        memcpy(&lo, ptr + 4, sizeof(uint32_t));
        hi = ntohl(hi);
        lo = ntohl(lo);
        v = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
    }
    return v;
}

void WriteU16(uint16_t value, uint8_t endian, uint8_t* out) {
    uint16_t v = value;
    if (endian == 2) {
        v = htons(v);
    }
    memcpy(out, &v, sizeof(uint16_t));
}

void WriteU32(uint32_t value, uint8_t endian, uint8_t* out) {
    uint32_t v = value;
    if (endian == 2) {
        v = htonl(v);
    }
    memcpy(out, &v, sizeof(uint32_t));
}

void WriteU64(uint64_t value, uint8_t endian, uint8_t* out) {
    if (endian == 2) {
        uint32_t hi = htonl(static_cast<uint32_t>(value >> 32));
        uint32_t lo = htonl(static_cast<uint32_t>(value & 0xFFFFFFFFu));
        memcpy(out, &hi, sizeof(uint32_t));
        memcpy(out + 4, &lo, sizeof(uint32_t));
    } else {
        memcpy(out, &value, sizeof(uint64_t));
    }
}

struct FrameBufferEntry {
    AudioPacketHeader referenceHeader{};
    uint8_t endian = 1;
    uint8_t fecK = 0;
    uint8_t fecM = 0;
    uint16_t shardSize = 0;
    uint32_t originalBytes = 0;
    uint32_t streamId = 0;
    uint32_t frameId = 0;
    uint64_t firstShardTimeMs = 0;
    std::map<uint32_t, std::vector<uint8_t>> shards;
};

struct DecodedFrame {
    std::vector<uint8_t> pcm;
    AudioPacketHeader referenceHeader{};
};

std::atomic<bool> g_audioRunning{false};
std::thread g_audioThread;

std::mutex g_frameMutex;
std::unordered_map<uint64_t, FrameBufferEntry> g_frameBuffers;
std::map<uint32_t, DecodedFrame> g_readyFrames;

std::optional<uint32_t> g_currentStreamId;
uint32_t g_expectedFrameId = 0;

constexpr uint16_t kAudioHeaderSize = 52;
constexpr uint8_t kExpectedVersion = 1;
constexpr uint16_t kTimeoutMs = 40;

uint64_t MakeFrameKey(uint32_t streamId, uint32_t frameId) {
    return (static_cast<uint64_t>(streamId) << 32) | frameId;
}

void ResetStream(uint32_t streamId, uint32_t frameId) {
    std::lock_guard<std::mutex> lock(g_frameMutex);
    g_frameBuffers.clear();
    g_readyFrames.clear();
    g_currentStreamId = streamId;
    g_expectedFrameId = frameId;
    DebugLog(L"AudioReceiver: Stream reset to streamId=" + std::to_wstring(streamId) +
             L" frameId=" + std::to_wstring(frameId));
}

std::string HexDump(const std::vector<uint8_t>& data) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (size_t i = 0; i < data.size(); ++i) {
        oss << std::setw(2) << static_cast<int>(data[i]);
        if ((i + 1) % 32 == 0) {
            oss << '\n';
        } else if ((i + 1) % 2 == 0) {
            oss << ' ';
        }
    }
    return oss.str();
}

void DumpDecodedFrame(uint32_t frameId, uint8_t fecK, uint8_t fecM, const AudioPacketHeader& hdr,
                      const std::vector<uint8_t>& pcm) {
    AudioPacketHeader logHeader = hdr;
    logHeader.shardIndex = 0;
    logHeader.flags &= ~static_cast<uint16_t>(1); // Clear IS_PARITY
    logHeader.shardSize = static_cast<uint16_t>(pcm.size());
    logHeader.originalBytes = static_cast<uint32_t>(pcm.size());
    logHeader.fecK = fecK;
    logHeader.fecM = fecM;

    // Compute CRC32 over header(with crc32=0) + PCM
    uint8_t headerBytes[sizeof(AudioPacketHeader)];
    memcpy(headerBytes, &logHeader, sizeof(AudioPacketHeader));
    WriteU32(0, logHeader.endian, reinterpret_cast<uint8_t*>(&headerBytes[48]));

    std::vector<uint8_t> blob;
    blob.insert(blob.end(), headerBytes, headerBytes + sizeof(AudioPacketHeader));
    blob.insert(blob.end(), pcm.begin(), pcm.end());

    uint32_t crc = ComputeCrc32(blob.data(), blob.size());
    WriteU32(crc, logHeader.endian, reinterpret_cast<uint8_t*>(&headerBytes[48]));

    // Rebuild blob with correct CRC
    blob.clear();
    blob.insert(blob.end(), headerBytes, headerBytes + sizeof(AudioPacketHeader));
    blob.insert(blob.end(), pcm.begin(), pcm.end());

    std::wstring logMessage = L"==== Audio Frame " + std::to_wstring(frameId) +
                              L" (K=" + std::to_wstring(fecK) + L" M=" + std::to_wstring(fecM) + L") ===\n";
    std::string hex = HexDump(blob);
    logMessage += std::wstring(hex.begin(), hex.end());
    DebugLog(logMessage);
}

void EnqueueDecoded(uint32_t streamId, uint32_t frameId, const FrameBufferEntry& fb, std::vector<uint8_t> pcm) {
    DecodedFrame decoded;
    decoded.pcm = std::move(pcm);
    decoded.referenceHeader = fb.referenceHeader;
    {
        std::lock_guard<std::mutex> lock(g_frameMutex);
        g_readyFrames.emplace(frameId, std::move(decoded));

        while (!g_readyFrames.empty()) {
            auto it = g_readyFrames.find(g_expectedFrameId);
            if (it == g_readyFrames.end()) break;

            DumpDecodedFrame(g_expectedFrameId, fb.fecK, fb.fecM, it->second.referenceHeader, it->second.pcm);
            // Placeholder for playback integration with WASAPI Shared
            // In this environment we only log the decoded data.
            g_readyFrames.erase(it);
            ++g_expectedFrameId;
        }
    }
}

void HandleTimeouts() {
    const uint64_t nowMs = SteadyNowMs();
    std::vector<FrameBufferEntry> timedOut;
    {
        std::lock_guard<std::mutex> lock(g_frameMutex);
        for (auto it = g_frameBuffers.begin(); it != g_frameBuffers.end();) {
            if (it->second.firstShardTimeMs > 0 &&
                nowMs - it->second.firstShardTimeMs >= kTimeoutMs &&
                it->second.shards.size() < static_cast<size_t>(it->second.fecK)) {
                timedOut.emplace_back(it->second);
                it = g_frameBuffers.erase(it);
            } else {
                ++it;
            }
        }
    }

    for (auto& fb : timedOut) {
        std::vector<uint8_t> silence(fb.originalBytes, 0);
        EnqueueDecoded(fb.streamId, fb.frameId, fb, std::move(silence));
    }
}

bool ValidateAndParseHeader(const uint8_t* packet, size_t len, AudioPacketHeader& outHeader) {
    if (len < kAudioHeaderSize) {
        DebugLog(L"AudioReceiver: Packet too small");
        return false;
    }

    memcpy(&outHeader, packet, sizeof(AudioPacketHeader));
    uint8_t endian = outHeader.endian;

    if (memcmp(outHeader.magic, "AUD0", 4) != 0) {
        DebugLog(L"AudioReceiver: Invalid magic");
        return false;
    }
    if (outHeader.version != kExpectedVersion) {
        DebugLog(L"AudioReceiver: Unsupported version");
        return false;
    }
    if (ReadU16(reinterpret_cast<const uint8_t*>(&outHeader.headerSize), endian) != kAudioHeaderSize) {
        DebugLog(L"AudioReceiver: Invalid header size");
        return false;
    }
    if (ReadU16(reinterpret_cast<const uint8_t*>(&outHeader.reserved), endian) != 0) {
        DebugLog(L"AudioReceiver: Reserved field mismatch");
        return false;
    }

    const uint16_t shardSize = ReadU16(reinterpret_cast<const uint8_t*>(&outHeader.shardSize), endian);
    if (len != kAudioHeaderSize + shardSize) {
        DebugLog(L"AudioReceiver: Payload length mismatch");
        return false;
    }

    // CRC32 check
    std::vector<uint8_t> temp(packet, packet + len);
    WriteU32(0, endian, &temp[48]);
    uint32_t crc = ComputeCrc32(temp.data(), temp.size());
    uint32_t packetCrc = ReadU32(reinterpret_cast<const uint8_t*>(&outHeader.crc32), endian);
    if (crc != packetCrc) {
        DebugLog(L"AudioReceiver: CRC mismatch");
        return false;
    }

    return true;
}

void ProcessPacket(const uint8_t* packet, size_t len) {
    AudioPacketHeader header{};
    if (!ValidateAndParseHeader(packet, len, header)) {
        return;
    }
    uint8_t endian = header.endian;

    uint32_t streamId = ReadU32(reinterpret_cast<const uint8_t*>(&header.streamId), endian);
    uint32_t frameId = ReadU32(reinterpret_cast<const uint8_t*>(&header.frameId), endian);
    uint16_t shardIndex = ReadU16(reinterpret_cast<const uint8_t*>(&header.shardIndex), endian);
    uint8_t fecK = header.fecK;
    uint8_t fecM = header.fecM;
    uint16_t shardSize = ReadU16(reinterpret_cast<const uint8_t*>(&header.shardSize), endian);
    uint32_t originalBytes = ReadU32(reinterpret_cast<const uint8_t*>(&header.originalBytes), endian);

    if (!g_currentStreamId.has_value() || g_currentStreamId.value() != streamId) {
        ResetStream(streamId, frameId);
    }

    uint64_t key = MakeFrameKey(streamId, frameId);
    FrameBufferEntry* fb = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_frameMutex);
        FrameBufferEntry& entry = g_frameBuffers[key];
        if (entry.firstShardTimeMs == 0) {
            entry.referenceHeader = header;
            entry.endian = endian;
            entry.fecK = fecK;
            entry.fecM = fecM;
            entry.shardSize = shardSize;
            entry.originalBytes = originalBytes;
            entry.streamId = streamId;
            entry.frameId = frameId;
            entry.firstShardTimeMs = SteadyNowMs();
        }

        auto insertResult = entry.shards.emplace(shardIndex, std::vector<uint8_t>(packet + kAudioHeaderSize, packet + len));
        if (!insertResult.second) {
            insertResult.first->second.assign(packet + kAudioHeaderSize, packet + len);
        }
        fb = &entry;
    }

    if (!fb) return;

    if (fb->shards.size() >= static_cast<size_t>(fb->fecK)) {
        std::map<uint32_t, std::vector<uint8_t>> shardsCopy;
        {
            std::lock_guard<std::mutex> lock(g_frameMutex);
            auto it = g_frameBuffers.find(key);
            if (it != g_frameBuffers.end()) {
                shardsCopy = it->second.shards;
                g_frameBuffers.erase(it);
            }
        }

        std::vector<uint8_t> decoded;
        if (!DecodeFEC_ISAL(shardsCopy, fb->fecK, fb->fecM, fb->originalBytes, decoded)) {
            decoded.assign(fb->originalBytes, 0);
        }
        EnqueueDecoded(streamId, frameId, *fb, std::move(decoded));
    }
}

void AudioThread() {
    DebugLog(L"AudioReceiver: thread started");

    SOCKET sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock == INVALID_SOCKET) {
        DebugLog(L"AudioReceiver: failed to create socket");
        return;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(8200);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        DebugLog(L"AudioReceiver: bind failed");
        closesocket(sock);
        return;
    }

    constexpr size_t kMaxPacket = 65535;
    std::vector<uint8_t> buffer(kMaxPacket);

    while (g_audioRunning.load(std::memory_order_acquire)) {
        fd_set readSet;
        FD_ZERO(&readSet);
        FD_SET(sock, &readSet);
        timeval tv{};
        tv.tv_sec = 0;
        tv.tv_usec = 10 * 1000; // 10ms

        int selectRes = select(0, &readSet, nullptr, nullptr, &tv);
        if (selectRes == SOCKET_ERROR) {
            continue;
        }
        if (selectRes == 0) {
            HandleTimeouts();
            continue;
        }

        sockaddr_in remoteAddr{};
        int addrLen = sizeof(remoteAddr);
        int received = recvfrom(sock, reinterpret_cast<char*>(buffer.data()), static_cast<int>(buffer.size()), 0,
                                reinterpret_cast<sockaddr*>(&remoteAddr), &addrLen);
        if (received <= 0) {
            continue;
        }

        ProcessPacket(buffer.data(), static_cast<size_t>(received));
        HandleTimeouts();
    }

    closesocket(sock);
    DebugLog(L"AudioReceiver: thread stopped");
}

}  // namespace

void StartAudioReceiver() {
    bool expected = false;
    if (g_audioRunning.compare_exchange_strong(expected, true)) {
        g_audioThread = std::thread(AudioThread);
    }
}

void StopAudioReceiver() {
    bool expected = true;
    if (g_audioRunning.compare_exchange_strong(expected, false)) {
        if (g_audioThread.joinable()) {
            g_audioThread.join();
        }
    }
}

