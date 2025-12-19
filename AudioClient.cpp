#include "AudioClient.h"
#include "DebugLog.h"
#include "ReedSolomon.h"

#include <Audioclient.h>
#include <mmdeviceapi.h>
#include <avrt.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <wrl/client.h>
#include <Windows.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iomanip>
#include <map>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#pragma comment(lib, "Mmdevapi.lib")
#pragma comment(lib, "Avrt.lib")

namespace {
constexpr uint16_t kPcmBytesInBlock = 1920;
constexpr uint16_t kBlockSamplesPerCh = 480;
constexpr uint16_t kSampleRate = 48000;
constexpr uint16_t kChannels = 2;
constexpr uint16_t kBitsPerSample = 16;
constexpr uint16_t kBytesPerFrame = kChannels * (kBitsPerSample / 8);
constexpr uint32_t kReceivePort = 8200;
constexpr size_t kHeaderSize = sizeof(AudioPacketHeader);
constexpr size_t kJitterBlocks = 2;
constexpr std::chrono::milliseconds kBlockTimeoutMs(15);

uint32_t ComputeCrc32(const uint8_t* data, size_t length) {
    static uint32_t table[256];
    static bool initialized = false;
    if (!initialized) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t crc = i;
            for (int j = 0; j < 8; ++j) {
                if (crc & 1) crc = (crc >> 1) ^ 0xEDB88320u;
                else crc >>= 1;
            }
            table[i] = crc;
        }
        initialized = true;
    }

    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < length; ++i) {
        uint8_t idx = static_cast<uint8_t>((crc ^ data[i]) & 0xFF);
        crc = (crc >> 8) ^ table[idx];
    }
    return crc ^ 0xFFFFFFFFu;
}

std::wstring HexDump(const std::vector<uint8_t>& buffer) {
    std::wostringstream oss;
    oss << std::hex << std::uppercase << std::setfill(L'0');
    for (size_t i = 0; i < buffer.size(); ++i) {
        oss << std::setw(2) << static_cast<int>(buffer[i]);
        if (i + 1 < buffer.size()) oss << L" ";
    }
    return oss.str();
}

struct BlockKey {
    uint32_t sessionId;
    uint32_t blockId;

    bool operator==(const BlockKey& other) const noexcept {
        return sessionId == other.sessionId && blockId == other.blockId;
    }
};

struct BlockKeyHash {
    size_t operator()(const BlockKey& k) const noexcept {
        return (static_cast<size_t>(k.sessionId) << 32) ^ k.blockId;
    }
};

struct BlockAssembly {
    uint16_t k = 0;
    uint16_t m = 0;
    uint16_t shardBytes = 0;
    uint16_t shardTotal = 0;
    uint16_t pcmBytes = 0;
    uint32_t blockCrc32 = 0;
    uint64_t captureTimestampNs = 0;
    std::chrono::steady_clock::time_point firstSeen;
    std::vector<std::optional<std::vector<uint8_t>>> shards;
    size_t shardCount = 0;
};

class AudioJitterBuffer {
public:
    AudioJitterBuffer(size_t depth, std::chrono::milliseconds timeout)
        : jitterDepth_(depth), timeout_(timeout) {}

    void Push(uint32_t sessionId, uint32_t blockId, const std::vector<uint8_t>& pcm) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!hasSession_ || sessionId != sessionId_) {
            sessionId_ = sessionId;
            hasSession_ = true;
            expectedBlockId_ = blockId;
            started_ = false;
            pendingMissing_.reset();
            blocks_.clear();
        }
        blocks_[blockId] = pcm;
        cv_.notify_all();
    }

    bool Pop(std::vector<uint8_t>& out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!hasSession_) return false;

        if (!started_) {
            if (blocks_.size() >= jitterDepth_) {
                started_ = true;
            } else {
                return false;
            }
        }

        auto it = blocks_.find(expectedBlockId_);
        if (it != blocks_.end()) {
            out = it->second;
            blocks_.erase(it);
            ++expectedBlockId_;
            pendingMissing_.reset();
            return true;
        }

        auto now = std::chrono::steady_clock::now();
        if (!pendingMissing_.has_value()) {
            pendingMissing_ = now;
            return false;
        }
        if (now - *pendingMissing_ >= timeout_) {
            out.assign(kPcmBytesInBlock, 0);
            ++expectedBlockId_;
            pendingMissing_ = now;
            return true;
        }
        return false;
    }

    void WaitForData() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, std::chrono::milliseconds(5));
    }

private:
    size_t jitterDepth_;
    std::chrono::milliseconds timeout_;
    bool hasSession_ = false;
    bool started_ = false;
    uint32_t sessionId_ = 0;
    uint32_t expectedBlockId_ = 0;
    std::optional<std::chrono::steady_clock::time_point> pendingMissing_;
    std::unordered_map<uint32_t, std::vector<uint8_t>> blocks_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

AudioJitterBuffer g_jitterBuffer(kJitterBlocks, kBlockTimeoutMs);
std::mutex g_blockMutex;
std::unordered_map<BlockKey, BlockAssembly, BlockKeyHash> g_blocks;

bool ValidateAndParse(const std::vector<uint8_t>& packet, AudioPacketHeader& header) {
    if (packet.size() < kHeaderSize) return false;
    std::memcpy(&header, packet.data(), kHeaderSize);

    if (std::strncmp(header.Magic, "AUDP", 4) != 0) return false;
    if (header.Version != 1 || header.HeaderSize != kHeaderSize) return false;

    std::vector<uint8_t> headerBytes(packet.begin(), packet.begin() + kHeaderSize);
    headerBytes[56] = headerBytes[57] = headerBytes[58] = headerBytes[59] = 0;
    uint32_t calcHeader = ComputeCrc32(headerBytes.data(), headerBytes.size());
    if (calcHeader != header.HeaderCrc32) return false;

    if (header.ShardBytes == 0) return false;
    const size_t payloadSize = packet.size() - kHeaderSize;
    if (payloadSize != header.ShardBytes) return false;

    uint32_t calcPayload = ComputeCrc32(packet.data() + kHeaderSize, payloadSize);
    if (calcPayload != header.PayloadCrc32) return false;

    if (header.ShardTotal != header.K + header.M) return false;
    if (header.ShardIndex >= header.ShardTotal) return false;
    if (header.PcmBytesInBlock != kPcmBytesInBlock || header.BlockSamplesPerCh != kBlockSamplesPerCh) return false;
    if (header.SampleRate != kSampleRate || header.Channels != kChannels || header.BitsPerSample != kBitsPerSample) return false;

    return true;
}

void LogDecodedBlock(const AudioPacketHeader& baseHeader, const std::vector<uint8_t>& pcm) {
    AudioPacketHeader logHeader = baseHeader;
    logHeader.Flags &= static_cast<uint8_t>(~0x01);
    logHeader.ShardIndex = 0;
    logHeader.ShardBytes = kPcmBytesInBlock;
    logHeader.PayloadCrc32 = ComputeCrc32(pcm.data(), pcm.size());

    std::vector<uint8_t> headerBytes(reinterpret_cast<uint8_t*>(&logHeader), reinterpret_cast<uint8_t*>(&logHeader) + kHeaderSize);
    headerBytes[56] = headerBytes[57] = headerBytes[58] = headerBytes[59] = 0;
    logHeader.HeaderCrc32 = ComputeCrc32(headerBytes.data(), headerBytes.size());
    std::memcpy(headerBytes.data() + 56, &logHeader.HeaderCrc32, sizeof(uint32_t));

    std::vector<uint8_t> combined;
    combined.reserve(kHeaderSize + pcm.size());
    combined.insert(combined.end(), headerBytes.begin(), headerBytes.end());
    combined.insert(combined.end(), pcm.begin(), pcm.end());

    DebugLog(HexDump(combined));
}

void HandleShard(const AudioPacketHeader& header, const std::vector<uint8_t>& payload) {
    BlockKey key{header.SessionId, header.BlockId};
    std::lock_guard<std::mutex> lock(g_blockMutex);
    auto& blk = g_blocks[key];
    if (blk.shards.empty()) {
        blk.k = header.K;
        blk.m = header.M;
        blk.shardBytes = header.ShardBytes;
        blk.shardTotal = header.ShardTotal;
        blk.pcmBytes = header.PcmBytesInBlock;
        blk.blockCrc32 = header.BlockCrc32;
        blk.captureTimestampNs = header.CaptureTimestampNs;
        blk.firstSeen = std::chrono::steady_clock::now();
        blk.shards.resize(header.ShardTotal);
    }

    if (header.ShardIndex >= blk.shards.size()) return;
    if (!blk.shards[header.ShardIndex].has_value()) {
        blk.shards[header.ShardIndex] = payload;
        ++blk.shardCount;
    }

    if (blk.shardCount < blk.k) return;

    std::map<uint32_t, std::vector<uint8_t>> shardMap;
    for (uint32_t i = 0; i < blk.shards.size(); ++i) {
        if (blk.shards[i].has_value()) {
            shardMap[i] = *blk.shards[i];
        }
    }

    std::vector<uint8_t> decoded;
    if (!DecodeFEC_ISAL(shardMap, blk.k, blk.m, blk.pcmBytes, decoded)) {
        return;
    }

    uint32_t calcBlock = ComputeCrc32(decoded.data(), decoded.size());
    if (calcBlock != blk.blockCrc32) {
        return;
    }

    LogDecodedBlock(header, decoded);
    g_jitterBuffer.Push(header.SessionId, header.BlockId, decoded);
    g_blocks.erase(key);
}

void AudioReceiverThread() {
    SOCKET sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock == INVALID_SOCKET) {
        DebugLog(L"AudioReceiver: failed to create socket");
        return;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(kReceivePort);

    if (bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        DebugLog(L"AudioReceiver: bind failed");
        closesocket(sock);
        return;
    }

    DWORD timeoutMs = 50;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeoutMs), sizeof(timeoutMs));

    std::vector<uint8_t> buffer(65536);
    while (g_audioReceiverRunning.load(std::memory_order_relaxed)) {
        int recvLen = recv(sock, reinterpret_cast<char*>(buffer.data()), static_cast<int>(buffer.size()), 0);
        if (recvLen <= 0) {
            continue;
        }
        buffer.resize(static_cast<size_t>(recvLen));
        AudioPacketHeader header{};
        if (!ValidateAndParse(buffer, header)) {
            buffer.resize(65536);
            continue;
        }
        std::vector<uint8_t> payload(buffer.begin() + kHeaderSize, buffer.end());
        HandleShard(header, payload);
        buffer.resize(65536);
    }

    closesocket(sock);
    DebugLog(L"AudioReceiver: stopped");
}

class WasapiRenderer {
public:
    bool Init() {
        HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (SUCCEEDED(hr)) {
            comInitialized_ = true;
        } else if (hr != RPC_E_CHANGED_MODE) {
            DebugLog(L"AudioPlayback: CoInitializeEx failed");
            return false;
        }

        hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, IID_PPV_ARGS(&deviceEnumerator_));
        if (FAILED(hr)) {
            DebugLog(L"AudioPlayback: failed to create MMDeviceEnumerator");
            return false;
        }

        hr = deviceEnumerator_->GetDefaultAudioEndpoint(eRender, eConsole, &device_);
        if (FAILED(hr)) {
            DebugLog(L"AudioPlayback: failed to get default endpoint");
            return false;
        }

        hr = device_->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, reinterpret_cast<void**>(&audioClient_));
        if (FAILED(hr)) {
            DebugLog(L"AudioPlayback: failed to activate IAudioClient");
            return false;
        }

        WAVEFORMATEXTENSIBLE format{};
        format.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
        format.Format.nChannels = kChannels;
        format.Format.nSamplesPerSec = kSampleRate;
        format.Format.wBitsPerSample = kBitsPerSample;
        format.Format.nBlockAlign = (kChannels * kBitsPerSample) / 8;
        format.Format.nAvgBytesPerSec = format.Format.nSamplesPerSec * format.Format.nBlockAlign;
        format.Format.cbSize = sizeof(WAVEFORMATEXTENSIBLE) - sizeof(WAVEFORMATEX);
        format.Samples.wValidBitsPerSample = kBitsPerSample;
        format.dwChannelMask = SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT;
        format.SubFormat = KSDATAFORMAT_SUBTYPE_PCM;

        REFERENCE_TIME bufferDuration = 20 * 10000; // 20ms
        hr = audioClient_->Initialize(AUDCLNT_SHAREMODE_SHARED,
                                      AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                                      bufferDuration,
                                      0,
                                      &format.Format,
                                      nullptr);
        if (FAILED(hr)) {
            DebugLog(L"AudioPlayback: IAudioClient Initialize failed");
            return false;
        }

        hr = audioClient_->GetBufferSize(&bufferFrameCount_);
        if (FAILED(hr)) {
            DebugLog(L"AudioPlayback: GetBufferSize failed");
            return false;
        }

        eventHandle_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (!eventHandle_) {
            DebugLog(L"AudioPlayback: CreateEvent failed");
            return false;
        }

        hr = audioClient_->SetEventHandle(eventHandle_);
        if (FAILED(hr)) {
            DebugLog(L"AudioPlayback: SetEventHandle failed");
            return false;
        }

        hr = audioClient_->GetService(IID_PPV_ARGS(&renderClient_));
        if (FAILED(hr)) {
            DebugLog(L"AudioPlayback: GetService failed");
            return false;
        }

        hr = audioClient_->Start();
        if (FAILED(hr)) {
            DebugLog(L"AudioPlayback: Start failed");
            return false;
        }
        return true;
    }

    void Shutdown() {
        if (audioClient_) {
            audioClient_->Stop();
        }
        if (eventHandle_) {
            CloseHandle(eventHandle_);
            eventHandle_ = nullptr;
        }
        renderClient_.Reset();
        audioClient_.Reset();
        device_.Reset();
        deviceEnumerator_.Reset();
        if (comInitialized_) {
            CoUninitialize();
        }
    }

    void Run() {
        std::vector<uint8_t> stash;
        stash.reserve(kPcmBytesInBlock);

        while (g_audioPlaybackRunning.load(std::memory_order_relaxed)) {
            DWORD waitRes = WaitForSingleObject(eventHandle_, 20);
            if (waitRes != WAIT_OBJECT_0 && waitRes != WAIT_TIMEOUT) {
                continue;
            }

            UINT32 padding = 0;
            if (FAILED(audioClient_->GetCurrentPadding(&padding))) {
                continue;
            }
            UINT32 availableFrames = bufferFrameCount_ - padding;
            while (availableFrames > 0) {
                UINT32 framesToWrite = availableFrames;
                BYTE* data = nullptr;
                if (FAILED(renderClient_->GetBuffer(framesToWrite, &data))) {
                    break;
                }

                size_t bytesNeeded = static_cast<size_t>(framesToWrite) * kBytesPerFrame;
                size_t written = 0;
                while (written < bytesNeeded) {
                    if (stash.empty()) {
                        std::vector<uint8_t> block;
                        if (!g_jitterBuffer.Pop(block)) {
                            block.assign(kPcmBytesInBlock, 0);
                        }
                        stash.insert(stash.end(), block.begin(), block.end());
                    }
                    size_t toCopy = std::min(bytesNeeded - written, stash.size());
                    std::memcpy(data + written, stash.data(), toCopy);
                    stash.erase(stash.begin(), stash.begin() + static_cast<std::ptrdiff_t>(toCopy));
                    written += toCopy;
                }

                renderClient_->ReleaseBuffer(framesToWrite, 0);
                if (availableFrames < framesToWrite) break;
                availableFrames -= framesToWrite;
            }

            g_jitterBuffer.WaitForData();
        }
    }

private:
    Microsoft::WRL::ComPtr<IMMDeviceEnumerator> deviceEnumerator_;
    Microsoft::WRL::ComPtr<IMMDevice> device_;
    Microsoft::WRL::ComPtr<IAudioClient> audioClient_;
    Microsoft::WRL::ComPtr<IAudioRenderClient> renderClient_;
    HANDLE eventHandle_ = nullptr;
    UINT32 bufferFrameCount_ = 0;
    bool comInitialized_ = false;
};

void AudioPlaybackThread() {
    WasapiRenderer renderer;
    if (!renderer.Init()) {
        DebugLog(L"AudioPlayback: init failed");
        return;
    }
    renderer.Run();
    renderer.Shutdown();
    DebugLog(L"AudioPlayback: stopped");
}

} // namespace

std::atomic<bool> g_audioReceiverRunning{true};
std::atomic<bool> g_audioPlaybackRunning{true};

bool StartAudioClient(std::thread& receiverThread, std::thread& playbackThread) {
    g_audioReceiverRunning.store(true, std::memory_order_relaxed);
    g_audioPlaybackRunning.store(true, std::memory_order_relaxed);

    try {
        receiverThread = std::thread(AudioReceiverThread);
        playbackThread = std::thread(AudioPlaybackThread);
    } catch (...) {
        DebugLog(L"StartAudioClient: failed to create threads");
        return false;
    }
    return true;
}

void StopAudioClient() {
    g_audioReceiverRunning.store(false, std::memory_order_relaxed);
    g_audioPlaybackRunning.store(false, std::memory_order_relaxed);
}
