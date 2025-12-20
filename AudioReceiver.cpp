#include "AudioReceiver.h"
#include "DebugLog.h"
#include "ReedSolomon.h"

#include <winsock2.h>
#include <WS2tcpip.h>
#include <Windows.h>
#include <atlbase.h>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <sstream>

namespace {
constexpr uint16_t kUdpPort = 8200;
constexpr size_t kAudioHeaderSize = 52;
constexpr uint32_t kSampleRate = 48000;
constexpr uint16_t kChannels = 2;
constexpr uint16_t kBitsPerSample = 16;
constexpr std::chrono::milliseconds kTimeoutMs(40);
constexpr size_t kMaxBufferedFrames = 10;

uint16_t ReadUint16(const uint8_t* p, uint8_t endian) {
    uint16_t v;
    std::memcpy(&v, p, sizeof(v));
    if (endian == 1) {
        return v;
    }
    return ntohs(v);
}

uint32_t ReadUint32(const uint8_t* p, uint8_t endian) {
    uint32_t v;
    std::memcpy(&v, p, sizeof(v));
    if (endian == 1) {
        return v;
    }
    return ntohl(v);
}

uint64_t ReadUint64(const uint8_t* p, uint8_t endian) {
    uint64_t v;
    std::memcpy(&v, p, sizeof(v));
    if (endian == 1) {
        return v;
    }
    uint32_t hi, lo;
    std::memcpy(&hi, reinterpret_cast<const uint8_t*>(p), sizeof(uint32_t));
    std::memcpy(&lo, reinterpret_cast<const uint8_t*>(p) + sizeof(uint32_t), sizeof(uint32_t));
    hi = ntohl(hi);
    lo = ntohl(lo);
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

void WriteUint16(uint8_t* p, uint16_t v, uint8_t endian) {
    if (endian == 1) {
        std::memcpy(p, &v, sizeof(v));
    } else {
        uint16_t be = htons(v);
        std::memcpy(p, &be, sizeof(be));
    }
}

void WriteUint32(uint8_t* p, uint32_t v, uint8_t endian) {
    if (endian == 1) {
        std::memcpy(p, &v, sizeof(v));
    } else {
        uint32_t be = htonl(v);
        std::memcpy(p, &be, sizeof(be));
    }
}

void WriteUint64(uint8_t* p, uint64_t v, uint8_t endian) {
    if (endian == 1) {
        std::memcpy(p, &v, sizeof(v));
        return;
    }
    uint32_t hi = static_cast<uint32_t>(v >> 32);
    uint32_t lo = static_cast<uint32_t>(v & 0xffffffffULL);
    hi = htonl(hi);
    lo = htonl(lo);
    std::memcpy(p, &hi, sizeof(uint32_t));
    std::memcpy(p + sizeof(uint32_t), &lo, sizeof(uint32_t));
}
}

struct AudioPacketHeader {
    char magic[4];
    uint8_t version = 0;
    uint8_t endian = 1;
    uint16_t headerSize = 0;
    uint16_t flags = 0;
    uint32_t streamId = 0;
    uint32_t frameId = 0;
    uint64_t captureTimeNs = 0;
    uint32_t sampleRate = 0;
    uint16_t channels = 0;
    uint16_t bitsPerSample = 0;
    uint16_t frameDurationUs = 0;
    uint16_t shardIndex = 0;
    uint8_t fecK = 0;
    uint8_t fecM = 0;
    uint16_t reserved = 0;
    uint16_t shardSize = 0;
    uint32_t originalBytes = 0;
    uint32_t crc32 = 0;
};

WasapiPlayer::WasapiPlayer() = default;
WasapiPlayer::~WasapiPlayer() { Stop(); }

bool WasapiPlayer::Initialize() {
    if (m_running.load()) {
        return true;
    }
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) {
        DebugLog(L"WasapiPlayer: CoInitializeEx failed.");
        return false;
    }

    CComPtr<IMMDeviceEnumerator> enumerator;
    hr = enumerator.CoCreateInstance(__uuidof(MMDeviceEnumerator));
    if (FAILED(hr)) {
        DebugLog(L"WasapiPlayer: failed to create device enumerator.");
        return false;
    }

    CComPtr<IMMDevice> device;
    hr = enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device);
    if (FAILED(hr)) {
        DebugLog(L"WasapiPlayer: failed to get default endpoint.");
        return false;
    }

    hr = device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, reinterpret_cast<void**>(&m_audioClient));
    if (FAILED(hr)) {
        DebugLog(L"WasapiPlayer: Activate failed.");
        return false;
    }

    WAVEFORMATEXTENSIBLE wfex = {};
    wfex.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
    wfex.Format.nChannels = kChannels;
    wfex.Format.nSamplesPerSec = kSampleRate;
    wfex.Format.wBitsPerSample = kBitsPerSample;
    wfex.Format.nBlockAlign = static_cast<WORD>((wfex.Format.nChannels * wfex.Format.wBitsPerSample) / 8);
    wfex.Format.nAvgBytesPerSec = wfex.Format.nSamplesPerSec * wfex.Format.nBlockAlign;
    wfex.Format.cbSize = sizeof(WAVEFORMATEXTENSIBLE) - sizeof(WAVEFORMATEX);
    wfex.Samples.wValidBitsPerSample = kBitsPerSample;
    wfex.dwChannelMask = SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT;
    wfex.SubFormat = KSDATAFORMAT_SUBTYPE_PCM;

    m_frameSizeBytes = wfex.Format.nBlockAlign;
    m_waveFormat = wfex.Format;

    const REFERENCE_TIME hnsBufferDuration = 10 * 1000 * 1000; // 1 second
    hr = m_audioClient->Initialize(
        AUDCLNT_SHAREMODE_SHARED,
        0,
        hnsBufferDuration,
        0,
        &wfex.Format,
        nullptr);
    if (FAILED(hr)) {
        DebugLog(L"WasapiPlayer: Initialize failed.");
        return false;
    }

    hr = m_audioClient->GetService(__uuidof(IAudioRenderClient), reinterpret_cast<void**>(&m_renderClient));
    if (FAILED(hr)) {
        DebugLog(L"WasapiPlayer: GetService failed.");
        return false;
    }

    hr = m_audioClient->GetBufferSize(&m_bufferFrameCount);
    if (FAILED(hr)) {
        DebugLog(L"WasapiPlayer: GetBufferSize failed.");
        return false;
    }

    hr = m_audioClient->Start();
    if (FAILED(hr)) {
        DebugLog(L"WasapiPlayer: Start failed.");
        return false;
    }

    m_running.store(true);
    m_thread = std::thread(&WasapiPlayer::PlaybackThread, this);
    return true;
}

void WasapiPlayer::ShutdownInternal() {
    if (m_audioClient) {
        m_audioClient->Stop();
    }
    m_renderClient.Release();
    m_audioClient.Release();
    m_running.store(false);
}

void WasapiPlayer::Stop() {
    m_running.store(false);
    m_queueCv.notify_all();
    if (m_thread.joinable()) {
        m_thread.join();
    }
    ShutdownInternal();
}

void WasapiPlayer::EnqueuePcm(const std::vector<uint8_t>& pcmData) {
    if (!m_running.load()) {
        if (!Initialize()) {
            return;
        }
    }
    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        m_queue.push(pcmData);
    }
    m_queueCv.notify_one();
}

void WasapiPlayer::PlaybackThread() {
    while (m_running.load()) {
        std::vector<uint8_t> data;
        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            m_queueCv.wait(lock, [&]() { return !m_running.load() || !m_queue.empty(); });
            if (!m_running.load() && m_queue.empty()) {
                break;
            }
            if (m_queue.empty()) {
                continue;
            }
            data = std::move(m_queue.front());
            m_queue.pop();
        }

        if (!m_audioClient || !m_renderClient) {
            continue;
        }

        UINT32 framesRemaining = static_cast<UINT32>(data.size() / m_frameSizeBytes);
        const uint8_t* src = data.data();
        while (framesRemaining > 0 && m_running.load()) {
            UINT32 padding = 0;
            if (FAILED(m_audioClient->GetCurrentPadding(&padding))) {
                break;
            }
            UINT32 framesAvailable = m_bufferFrameCount - padding;
            if (framesAvailable == 0) {
                Sleep(1);
                continue;
            }
            UINT32 toWrite = std::min(framesAvailable, framesRemaining);
            BYTE* buffer = nullptr;
            if (FAILED(m_renderClient->GetBuffer(toWrite, &buffer))) {
                break;
            }
            std::memcpy(buffer, src, toWrite * m_frameSizeBytes);
            m_renderClient->ReleaseBuffer(toWrite, 0);
            framesRemaining -= toWrite;
            src += toWrite * m_frameSizeBytes;
        }
    }
}

AudioReceiver::AudioReceiver() = default;
AudioReceiver::~AudioReceiver() { Stop(); }

bool AudioReceiver::InitializeSocket() {
    m_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (m_socket == INVALID_SOCKET) {
        DebugLog(L"AudioReceiver: socket creation failed.");
        return false;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(kUdpPort);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(m_socket, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        DebugLog(L"AudioReceiver: bind failed.");
        closesocket(m_socket);
        m_socket = INVALID_SOCKET;
        return false;
    }
    return true;
}

bool AudioReceiver::Start() {
    if (m_running.load()) {
        return true;
    }
    if (!InitializeSocket()) {
        return false;
    }
    if (!m_player.Initialize()) {
        return false;
    }
    m_running.store(true);
    m_receiverThread = std::thread(&AudioReceiver::ReceiverThread, this);
    m_timeoutThread = std::thread(&AudioReceiver::TimeoutThread, this);
    return true;
}

void AudioReceiver::Stop() {
    m_running.store(false);
    if (m_socket != INVALID_SOCKET) {
        closesocket(m_socket);
        m_socket = INVALID_SOCKET;
    }
    if (m_receiverThread.joinable()) m_receiverThread.join();
    if (m_timeoutThread.joinable()) m_timeoutThread.join();
    m_player.Stop();
}

void AudioReceiver::ReceiverThread() {
    DebugLog(L"AudioReceiver: Receiver thread started.");
    std::vector<uint8_t> buffer(65536);
    while (m_running.load()) {
        int received = recv(m_socket, reinterpret_cast<char*>(buffer.data()), static_cast<int>(buffer.size()), 0);
        if (received <= 0) {
            Sleep(1);
            continue;
        }
        if (received < static_cast<int>(kAudioHeaderSize)) {
            DebugLog(L"AudioReceiver: packet too small.");
            continue;
        }
        AudioPacketHeader hdr{};
        if (!ParseAudioPacketHeader(buffer.data(), static_cast<size_t>(received), hdr)) {
            continue;
        }
        size_t payloadLen = static_cast<size_t>(received) - kAudioHeaderSize;
        if (payloadLen != hdr.shardSize) {
            DebugLog(L"AudioReceiver: shardSize mismatch.");
            continue;
        }

        FrameKey key{hdr.streamId, hdr.frameId};
        std::vector<uint8_t> payload(payloadLen);
        std::memcpy(payload.data(), buffer.data() + kAudioHeaderSize, payloadLen);

        bool decodeNow = false;
        FrameBufferEntry entrySnapshot;
        {
            std::lock_guard<std::mutex> lock(m_frameMutex);
            if (!m_haveStream || hdr.streamId != m_currentStreamId) {
                m_frames.clear();
                m_pendingPlayback.clear();
                m_expectedFrameId = hdr.frameId;
                m_currentStreamId = hdr.streamId;
                m_haveStream = true;
            }

            auto& entry = m_frames[key];
            if (entry.shards.empty()) {
                entry.endian = hdr.endian;
                entry.flags = hdr.flags;
                entry.streamId = hdr.streamId;
                entry.frameId = hdr.frameId;
                entry.captureTimeNs = hdr.captureTimeNs;
                entry.sampleRate = hdr.sampleRate;
                entry.channels = hdr.channels;
                entry.bitsPerSample = hdr.bitsPerSample;
                entry.frameDurationUs = hdr.frameDurationUs;
                entry.shardSize = hdr.shardSize;
                entry.originalBytes = hdr.originalBytes;
                entry.fecK = hdr.fecK;
                entry.fecM = hdr.fecM;
                entry.firstShardTime = std::chrono::steady_clock::now();
            }
            if (entry.shards.find(hdr.shardIndex) == entry.shards.end()) {
                entry.shards[hdr.shardIndex] = std::move(payload);
            }

            if (entry.shards.size() >= hdr.fecK) {
                decodeNow = true;
                entrySnapshot = entry;
                m_frames.erase(key);
            }

            if (m_frames.size() > kMaxBufferedFrames) {
                m_frames.erase(m_frames.begin());
            }
        }

        if (decodeNow) {
            TryDecodeFrame(key, entrySnapshot);
        }
    }
    DebugLog(L"AudioReceiver: Receiver thread stopped.");
}

void AudioReceiver::TimeoutThread() {
    DebugLog(L"AudioReceiver: Timeout thread started.");
    while (m_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        std::vector<std::pair<FrameKey, FrameBufferEntry>> timedOut;
        auto now = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lock(m_frameMutex);
            for (auto it = m_frames.begin(); it != m_frames.end();) {
                if (it->second.firstShardTime.time_since_epoch().count() == 0) {
                    ++it;
                    continue;
                }
                if (now - it->second.firstShardTime >= kTimeoutMs && it->second.shards.size() < it->second.fecK) {
                    timedOut.emplace_back(it->first, it->second);
                    it = m_frames.erase(it);
                } else {
                    ++it;
                }
            }
        }
        for (const auto& pair : timedOut) {
            EnqueueSilence(pair.first, pair.second);
        }
    }
    DebugLog(L"AudioReceiver: Timeout thread stopped.");
}

bool AudioReceiver::TryDecodeFrame(const FrameKey& key, FrameBufferEntry entry) {
    (void)key;
    std::map<uint32_t, std::vector<uint8_t>> shards = entry.shards;
    std::vector<uint8_t> decoded;
    bool success = DecodeFEC_ISAL(shards, entry.fecK, entry.fecM, entry.originalBytes, decoded);
    if (!success || decoded.size() != entry.originalBytes) {
        DebugLog(L"AudioReceiver: FEC decode failed, using silence.");
        decoded.assign(entry.originalBytes, 0);
    }
    DecodedFrame frame{entry, std::move(decoded)};
    ProcessDecodedFrame(frame);
    return success;
}

void AudioReceiver::EnqueueSilence(const FrameKey& key, const FrameBufferEntry& entry) {
    FrameBufferEntry silentEntry = entry;
    silentEntry.flags &= ~0x1; // ensure parity flag cleared
    std::vector<uint8_t> silence(entry.originalBytes, 0);
    DecodedFrame frame{silentEntry, std::move(silence)};
    ProcessDecodedFrame(frame);
}

std::vector<uint8_t> BuildDecodedLogHeader(const AudioReceiver::FrameBufferEntry& info, const std::vector<uint8_t>& pcm) {
    std::vector<uint8_t> bytes(kAudioHeaderSize, 0);
    bytes[0] = 'A'; bytes[1] = 'U'; bytes[2] = 'D'; bytes[3] = '0';
    bytes[4] = 1;
    bytes[5] = info.endian;
    WriteUint16(bytes.data() + 6, static_cast<uint16_t>(kAudioHeaderSize), info.endian);
    WriteUint16(bytes.data() + 8, info.flags & ~0x1, info.endian);
    WriteUint32(bytes.data() + 10, info.streamId, info.endian);
    WriteUint32(bytes.data() + 14, info.frameId, info.endian);
    WriteUint64(bytes.data() + 18, info.captureTimeNs, info.endian);
    WriteUint32(bytes.data() + 26, info.sampleRate, info.endian);
    WriteUint16(bytes.data() + 30, info.channels, info.endian);
    WriteUint16(bytes.data() + 32, info.bitsPerSample, info.endian);
    WriteUint16(bytes.data() + 34, info.frameDurationUs, info.endian);
    WriteUint16(bytes.data() + 36, 0, info.endian);
    bytes[38] = info.fecK;
    bytes[39] = info.fecM;
    WriteUint16(bytes.data() + 40, 0, info.endian);
    WriteUint16(bytes.data() + 42, static_cast<uint16_t>(pcm.size()), info.endian);
    WriteUint32(bytes.data() + 44, static_cast<uint32_t>(pcm.size()), info.endian);
    WriteUint32(bytes.data() + 48, 0, info.endian);
    uint32_t crc = AudioCrc32(bytes.data(), kAudioHeaderSize);
    crc = AudioCrc32(pcm.data(), pcm.size(), crc);
    WriteUint32(bytes.data() + 48, crc, info.endian);
    return bytes;
}

void AudioReceiver::ProcessDecodedFrame(const DecodedFrame& frame) {
    FrameKey key{frame.headerInfo.streamId, frame.headerInfo.frameId};
    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        m_pendingPlayback[key] = frame;
    }
    PumpPlaybackQueue();
}

void AudioReceiver::PumpPlaybackQueue() {
    while (true) {
        DecodedFrame frame;
        FrameKey key{m_currentStreamId, m_expectedFrameId};
        {
            std::lock_guard<std::mutex> lock(m_frameMutex);
            auto it = m_pendingPlayback.find(key);
            if (it == m_pendingPlayback.end()) {
                break;
            }
            frame = it->second;
            m_pendingPlayback.erase(it);
        }

        auto headerBytes = BuildDecodedLogHeader(frame.headerInfo, frame.pcm);
        std::vector<uint8_t> blob;
        blob.reserve(headerBytes.size() + frame.pcm.size());
        blob.insert(blob.end(), headerBytes.begin(), headerBytes.end());
        blob.insert(blob.end(), frame.pcm.begin(), frame.pcm.end());

        std::wstringstream ss;
        ss << L"==== Audio Frame " << frame.headerInfo.frameId << L" K=" << static_cast<int>(frame.headerInfo.fecK)
           << L" M=" << static_cast<int>(frame.headerInfo.fecM) << L" ====";
        DebugLog(ss.str());
        DebugLog(HexDump(blob));

        m_player.EnqueuePcm(frame.pcm);
        ++m_expectedFrameId;
    }
}

bool ParseAudioPacketHeader(const uint8_t* data, size_t len, AudioPacketHeader& outHeader) {
    if (len < kAudioHeaderSize) {
        DebugLog(L"ParseAudioPacketHeader: len too small.");
        return false;
    }
    if (!(data[0] == 'A' && data[1] == 'U' && data[2] == 'D' && data[3] == '0')) {
        DebugLog(L"ParseAudioPacketHeader: magic mismatch.");
        return false;
    }
    outHeader.version = data[4];
    outHeader.endian = data[5];
    if (outHeader.version != 1) {
        DebugLog(L"ParseAudioPacketHeader: unsupported version.");
        return false;
    }
    outHeader.headerSize = ReadUint16(data + 6, outHeader.endian);
    if (outHeader.headerSize != kAudioHeaderSize) {
        DebugLog(L"ParseAudioPacketHeader: header size mismatch.");
        return false;
    }
    outHeader.flags = ReadUint16(data + 8, outHeader.endian);
    outHeader.streamId = ReadUint32(data + 10, outHeader.endian);
    outHeader.frameId = ReadUint32(data + 14, outHeader.endian);
    outHeader.captureTimeNs = ReadUint64(data + 18, outHeader.endian);
    outHeader.sampleRate = ReadUint32(data + 26, outHeader.endian);
    outHeader.channels = ReadUint16(data + 30, outHeader.endian);
    outHeader.bitsPerSample = ReadUint16(data + 32, outHeader.endian);
    outHeader.frameDurationUs = ReadUint16(data + 34, outHeader.endian);
    outHeader.shardIndex = ReadUint16(data + 36, outHeader.endian);
    outHeader.fecK = data[38];
    outHeader.fecM = data[39];
    outHeader.reserved = ReadUint16(data + 40, outHeader.endian);
    outHeader.shardSize = ReadUint16(data + 42, outHeader.endian);
    outHeader.originalBytes = ReadUint32(data + 44, outHeader.endian);
    outHeader.crc32 = ReadUint32(data + 48, outHeader.endian);

    if (outHeader.reserved != 0) {
        DebugLog(L"ParseAudioPacketHeader: reserved mismatch.");
        return false;
    }
    if (outHeader.shardSize + kAudioHeaderSize > len) {
        DebugLog(L"ParseAudioPacketHeader: shardSize exceeds packet size.");
        return false;
    }

    std::vector<uint8_t> headerCopy(data, data + kAudioHeaderSize);
    WriteUint32(headerCopy.data() + 48, 0, outHeader.endian);
    uint32_t crc = AudioCrc32(headerCopy.data(), headerCopy.size());
    crc = AudioCrc32(data + kAudioHeaderSize, outHeader.shardSize, crc);
    if (crc != outHeader.crc32) {
        DebugLog(L"ParseAudioPacketHeader: CRC mismatch.");
        return false;
    }
    return true;
}

uint32_t AudioCrc32(const uint8_t* data, size_t len) {
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int j = 0; j < 8; ++j) {
            uint32_t mask = -(crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

uint32_t AudioCrc32(const uint8_t* data, size_t len, uint32_t previous) {
    uint32_t crc = ~previous;
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int j = 0; j < 8; ++j) {
            uint32_t mask = -(crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

std::wstring HexDump(const std::vector<uint8_t>& data) {
    std::wostringstream oss;
    oss << std::hex << std::setfill(L'0');
    for (size_t i = 0; i < data.size(); ++i) {
        oss << std::setw(2) << static_cast<int>(data[i]);
        if ((i + 1) % 32 == 0) {
            oss << L"\n";
        }
    }
    return oss.str();
}
