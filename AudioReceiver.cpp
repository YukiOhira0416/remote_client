#include "AudioReceiver.h"
#include "ReedSolomon.h"
#include <audioclient.h>
#include <mmdeviceapi.h>
#include <avrt.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <sstream>

#pragma comment(lib, "Avrt.lib")

namespace {
constexpr uint16_t kBlockSamplesPerCh = 480;
constexpr uint16_t kPcmBytesInBlock = 1920;
constexpr uint16_t kSampleRate = 48000;
constexpr uint16_t kChannels = 2;
constexpr uint16_t kBitsPerSample = 16;
constexpr uint16_t kHeaderSize = 64;
constexpr uint8_t kVersion = 1;
constexpr uint32_t kBlockTimeoutMs = 15;

// CRC32 parameters (IEEE 802.3 reflected)
constexpr uint32_t kCrcPoly = 0xEDB88320u;

uint64_t NowMs() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

std::string ToHex(const uint8_t* data, size_t len) {
    std::ostringstream oss;
    oss << std::hex << std::uppercase << std::setfill('0');
    for (size_t i = 0; i < len; ++i) {
        oss << std::setw(2) << static_cast<int>(data[i]);
        if (i + 1 != len) oss << ' ';
    }
    return oss.str();
}
}

static_assert(sizeof(AudioReceiver::AudioPacketHeader) == 64, "AudioPacketHeader must be 64 bytes");

AudioReceiver::AudioReceiver() = default;
AudioReceiver::~AudioReceiver() { Stop(); }

bool AudioReceiver::Start() {
    if (running_.load()) return true;
    if (!InitializeSocket()) return false;
    if (!InitAudioClient()) {
        DebugLog(L"AudioReceiver: Failed to initialize WASAPI render client.");
        if (udpSocket_ != INVALID_SOCKET) {
            closesocket(udpSocket_);
            udpSocket_ = INVALID_SOCKET;
        }
        return false;
    }
    running_.store(true);
    recvThread_ = std::thread(&AudioReceiver::ReceiveLoop, this);
    renderThread_ = std::thread(&AudioReceiver::RenderLoop, this);
    return true;
}

void AudioReceiver::Stop() {
    if (!running_.exchange(false)) return;
    if (udpSocket_ != INVALID_SOCKET) {
        closesocket(udpSocket_);
        udpSocket_ = INVALID_SOCKET;
    }
    if (recvThread_.joinable()) recvThread_.join();
    {
        std::lock_guard lk(renderMutex_);
    }
    renderCv_.notify_all();
    if (renderThread_.joinable()) renderThread_.join();
    ShutdownAudioClient();
}

bool AudioReceiver::InitializeSocket() {
    udpSocket_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (udpSocket_ == INVALID_SOCKET) {
        DebugLog(L"AudioReceiver: Failed to create UDP socket.");
        return false;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(8200);
    if (bind(udpSocket_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        DebugLog(L"AudioReceiver: Failed to bind UDP socket.");
        closesocket(udpSocket_);
        udpSocket_ = INVALID_SOCKET;
        return false;
    }

    int bufSize = 1 << 20;
    setsockopt(udpSocket_, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<const char*>(&bufSize), sizeof(bufSize));
    return true;
}

uint32_t AudioReceiver::ComputeCrc32(const uint8_t* data, size_t len) const {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int b = 0; b < 8; ++b) {
            const uint32_t mask = -(crc & 1u);
            crc = (crc >> 1) ^ (kCrcPoly & mask);
        }
    }
    return crc ^ 0xFFFFFFFFu;
}

bool AudioReceiver::ParsePacket(const uint8_t* data, size_t len, ReceivedPacket& out) {
    if (len < kHeaderSize) return false;
    AudioPacketHeader header{};
    std::memcpy(&header, data, sizeof(AudioPacketHeader));

    if (std::memcmp(header.Magic, "AUDP", 4) != 0 || header.Version != kVersion || header.HeaderSize != kHeaderSize) {
        return false;
    }

    // Validate header CRC32
    AudioPacketHeader temp = header;
    temp.HeaderCrc32 = 0;
    uint32_t expectedHeaderCrc = ComputeCrc32(reinterpret_cast<uint8_t*>(&temp), sizeof(AudioPacketHeader));
    if (expectedHeaderCrc != header.HeaderCrc32) {
        return false;
    }

    if (header.ShardTotal != header.K + header.M) return false;
    if (header.ShardIndex >= header.ShardTotal) return false;
    if (header.PcmBytesInBlock != kPcmBytesInBlock) return false;
    if (header.BlockSamplesPerCh != kBlockSamplesPerCh) return false;
    if (header.SampleRate != kSampleRate || header.Channels != kChannels || header.BitsPerSample != kBitsPerSample) return false;

    const size_t payloadLen = len - kHeaderSize;
    if (payloadLen != header.ShardBytes) return false;
    if (ComputeCrc32(data + kHeaderSize, payloadLen) != header.PayloadCrc32) return false;

    out.header = header;
    out.payload.assign(data + kHeaderSize, data + len);
    out.arrival_ms = NowMs();
    return true;
}

void AudioReceiver::ReceiveLoop() {
    std::vector<uint8_t> buffer(2048);
    while (running_.load()) {
        int ret = recv(udpSocket_, reinterpret_cast<char*>(buffer.data()), static_cast<int>(buffer.size()), 0);
        if (ret <= 0) {
            if (!running_.load()) break;
            continue;
        }
        ReceivedPacket pkt{};
        if (!ParsePacket(buffer.data(), static_cast<size_t>(ret), pkt)) {
            continue;
        }
        TryAssemble(pkt);
    }
}

void AudioReceiver::TryAssemble(const ReceivedPacket& pkt) {
    BlockKey key{pkt.header.SessionId, pkt.header.BlockId};
    std::unique_lock lk(buffersMutex_);

    if (!haveExpectedBlock_ || currentSessionId_ != pkt.header.SessionId) {
        ResetSession(pkt.header.SessionId);
        expectedBlockId_ = pkt.header.BlockId;
        haveExpectedBlock_ = true;
    }

    auto& buffer = buffers_[key];
    if (buffer.firstSeenMs == 0) {
        buffer.firstSeenMs = pkt.arrival_ms;
        buffer.k = pkt.header.K;
        buffer.m = pkt.header.M;
        buffer.shardBytes = pkt.header.ShardBytes;
        buffer.pcmBytes = pkt.header.PcmBytesInBlock;
        buffer.shardTotal = pkt.header.ShardTotal;
        buffer.blockCrc32 = pkt.header.BlockCrc32;
        buffer.captureTimestampNs = pkt.header.CaptureTimestampNs;
    }

    if (pkt.header.ShardIndex < pkt.header.ShardTotal) {
        buffer.dataShards[pkt.header.ShardIndex] = pkt.payload;
    }

    if (buffer.dataShards.size() >= buffer.k) {
        BlockBuffer copy = buffer;
        lk.unlock();
        if (AttemptDecode(key, copy)) {
            std::lock_guard renderLock(renderMutex_);
            // store decoded block
        }
        return;
    }

    // Timeout handling
    if (pkt.arrival_ms - buffer.firstSeenMs > kBlockTimeoutMs) {
        BlockBuffer copy = buffer;
        lk.unlock();
        AttemptDecode(key, copy);
    }
}

bool AudioReceiver::AttemptDecode(BlockKey key, BlockBuffer& buffer) {
    std::map<uint32_t, std::vector<uint8_t>> shardMap;
    for (const auto& kv : buffer.dataShards) {
        shardMap[kv.first] = kv.second;
    }

    if (static_cast<int>(shardMap.size()) < buffer.k) return false;

    std::vector<uint8_t> decoded;
    if (!DecodeFEC_ISAL(shardMap, buffer.k, buffer.m, buffer.pcmBytes, decoded)) {
        return false;
    }
    if (decoded.size() < buffer.pcmBytes) return false;

    // Validate block CRC
    std::vector<uint8_t> pcm(decoded.begin(), decoded.begin() + buffer.pcmBytes);
    if (ComputeCrc32(pcm.data(), pcm.size()) != buffer.blockCrc32) {
        return false;
    }

    DecodedBlock block{};
    block.header = {};
    block.header.Magic[0] = 'A'; block.header.Magic[1] = 'U'; block.header.Magic[2] = 'D'; block.header.Magic[3] = 'P';
    block.header.Version = kVersion;
    block.header.HeaderSize = kHeaderSize;
    block.header.Flags = 0;
    if (block.header.ShardIndex == 0) block.header.Flags |= 0x02;
    if (block.header.ShardIndex + 1 == block.header.ShardTotal) block.header.Flags |= 0x04;
    block.header.SessionId = key.sessionId;
    block.header.BlockId = key.blockId;
    block.header.ShardIndex = 0;
    block.header.ShardTotal = buffer.shardTotal;
    block.header.K = buffer.k;
    block.header.M = buffer.m;
    block.header.ShardBytes = static_cast<uint16_t>(block.pcm.size());
    block.header.PcmBytesInBlock = buffer.pcmBytes;
    block.header.BlockSamplesPerCh = kBlockSamplesPerCh;
    block.header.SampleRate = kSampleRate;
    block.header.Channels = kChannels;
    block.header.BitsPerSample = kBitsPerSample;
    block.header.CaptureTimestampNs = buffer.captureTimestampNs;
    block.header.BlockCrc32 = buffer.blockCrc32;
    block.header.PayloadCrc32 = ComputeCrc32(pcm.data(), pcm.size());
    block.header.HeaderCrc32 = 0;
    block.header.HeaderCrc32 = ComputeCrc32(reinterpret_cast<uint8_t*>(&block.header), sizeof(AudioPacketHeader));
    block.pcm = std::move(pcm);
    block.completed_ms = NowMs();

    {
        std::lock_guard lk(renderMutex_);
        jitterBuffer_[key.blockId] = block;
    }
    renderCv_.notify_one();
    LogDecodedBlock(block);

    std::lock_guard lk(buffersMutex_);
    buffers_.erase(key);
    return true;
}

void AudioReceiver::LogDecodedBlock(const DecodedBlock& block) {
    std::vector<uint8_t> combined(sizeof(AudioPacketHeader) + block.pcm.size());
    std::memcpy(combined.data(), &block.header, sizeof(AudioPacketHeader));
    std::memcpy(combined.data() + sizeof(AudioPacketHeader), block.pcm.data(), block.pcm.size());
    std::string hex = ToHex(combined.data(), combined.size());
    DebugLog(L"AudioDecodedBlock " + std::wstring(hex.begin(), hex.end()));
}

bool AudioReceiver::InitAudioClient() {
    HRESULT hr = S_OK;
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    IMMDeviceEnumerator* enumerator = nullptr;
    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, IID_PPV_ARGS(&enumerator));
    if (FAILED(hr)) return false;

    hr = enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device_);
    enumerator->Release();
    if (FAILED(hr)) return false;

    hr = device_->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, reinterpret_cast<void**>(&audioClient_));
    if (FAILED(hr)) return false;

    hr = audioClient_->GetMixFormat(&mixFormat_);
    if (FAILED(hr)) return false;

    mixFormat_->wFormatTag = WAVE_FORMAT_PCM;
    mixFormat_->nChannels = kChannels;
    mixFormat_->nSamplesPerSec = kSampleRate;
    mixFormat_->wBitsPerSample = kBitsPerSample;
    mixFormat_->nBlockAlign = (mixFormat_->wBitsPerSample / 8) * mixFormat_->nChannels;
    mixFormat_->nAvgBytesPerSec = mixFormat_->nSamplesPerSec * mixFormat_->nBlockAlign;
    mixFormat_->cbSize = 0;

    REFERENCE_TIME defaultPeriod = 0;
    REFERENCE_TIME minPeriod = 0;
    device_->GetDevicePeriod(&defaultPeriod, &minPeriod);

    hr = audioClient_->Initialize(AUDCLNT_SHAREMODE_SHARED,
                                  AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                                  200000, 0, mixFormat_, nullptr);
    if (FAILED(hr)) return false;

    hr = audioClient_->GetBufferSize(&bufferFrameCount_);
    if (FAILED(hr)) return false;

    hr = audioClient_->GetService(IID_PPV_ARGS(&renderClient_));
    if (FAILED(hr)) return false;

    audioEvent_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!audioEvent_) return false;
    hr = audioClient_->SetEventHandle(audioEvent_);
    if (FAILED(hr)) return false;

    hr = audioClient_->Start();
    return SUCCEEDED(hr);
}

void AudioReceiver::ShutdownAudioClient() {
    if (audioClient_) {
        audioClient_->Stop();
    }
    if (audioEvent_) CloseHandle(audioEvent_);
    if (renderClient_) renderClient_->Release();
    if (audioClient_) audioClient_->Release();
    if (device_) device_->Release();
    if (mixFormat_) CoTaskMemFree(mixFormat_);
    audioEvent_ = nullptr;
    renderClient_ = nullptr;
    audioClient_ = nullptr;
    device_ = nullptr;
    mixFormat_ = nullptr;
    CoUninitialize();
}

bool AudioReceiver::WriteBlockToRenderClient(const uint8_t* data, size_t bytes) {
    if (!renderClient_ || !audioClient_) return false;
    UINT32 padding = 0;
    if (FAILED(audioClient_->GetCurrentPadding(&padding))) return false;

    const UINT32 framesAvailable = bufferFrameCount_ - padding;
    const UINT32 framesNeeded = static_cast<UINT32>(bytes / mixFormat_->nBlockAlign);
    if (framesAvailable < framesNeeded) {
        return false;
    }

    BYTE* buffer = nullptr;
    if (FAILED(renderClient_->GetBuffer(framesNeeded, &buffer))) return false;
    std::memcpy(buffer, data, bytes);
    renderClient_->ReleaseBuffer(framesNeeded, 0);
    return true;
}

void AudioReceiver::RenderLoop() {
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const size_t blockBytes = kPcmBytesInBlock;
    auto playSilence = [this, blockBytes]() {
        std::vector<uint8_t> zeros(blockBytes, 0);
        WriteBlockToRenderClient(zeros.data(), zeros.size());
    };

    HANDLE events[1];
    events[0] = audioEvent_;

    uint32_t jitteredStartBlocks = jitterBlocks_;
    while (running_.load()) {
        DWORD waitRes = WaitForMultipleObjects(1, events, FALSE, 20);
        if (waitRes == WAIT_OBJECT_0) {
            std::unique_lock lk(renderMutex_);
            renderCv_.wait_for(lk, std::chrono::milliseconds(1));

            if (!haveExpectedBlock_) {
                lk.unlock();
                playSilence();
                continue;
            }

            if (jitterBuffer_.size() < jitteredStartBlocks) {
                lk.unlock();
                playSilence();
                continue;
            }

            auto it = jitterBuffer_.find(expectedBlockId_);
            if (it != jitterBuffer_.end()) {
                auto pcm = it->second.pcm;
                jitterBuffer_.erase(it);
                expectedBlockId_ += 1;
                lk.unlock();
                if (!WriteBlockToRenderClient(pcm.data(), pcm.size())) {
                    playSilence();
                }
                continue;
            }

            // Timeout check
            uint64_t oldestMs = NowMs();
            if (!jitterBuffer_.empty()) {
                oldestMs = jitterBuffer_.begin()->second.completed_ms;
            }
            if (NowMs() - oldestMs > kBlockTimeoutMs) {
                expectedBlockId_ += 1;
                lk.unlock();
                playSilence();
                continue;
            }
            lk.unlock();
            playSilence();
        }
    }
    CoUninitialize();
}

void AudioReceiver::ResetSession(uint32_t sessionId) {
    buffers_.clear();
    jitterBuffer_.clear();
    currentSessionId_ = sessionId;
    haveExpectedBlock_ = false;
    expectedBlockId_ = 0;
}

