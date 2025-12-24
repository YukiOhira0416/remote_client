#include "AudioClient.h"

#include <Windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <avrt.h>
#include <mmsystem.h>
#include <algorithm>
#include <atlbase.h>
#include <chrono>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>

#include "DebugLog.h"
#include "Globals.h"
#include "ReedSolomon.h"
#include "TimeSyncClient.h"
#include "concurrentqueue/concurrentqueue.h"

#pragma comment(lib, "Mmdevapi.lib")
#pragma comment(lib, "Avrt.lib")

namespace {

// ===== Packet structures =====
enum class AudioPayloadType : uint8_t {
    AUDIO_SHARD = 0x01,
    AUDIO_FORMAT = 0x02
};

#pragma pack(push, 1)
struct AudioUdpHeader {
    uint8_t header_bytes;   // Total header size including this byte
    uint8_t payload_type;   // AudioPayloadType
    uint16_t reserved = 0;  // align to 4 bytes, reserved for future use
};

struct AudioShardHeader {
    uint32_t block_id_net;          // network byte order
    uint16_t shard_index_net;       // network byte order
    uint16_t k_net;                 // network byte order
    uint16_t m_net;                 // network byte order
    uint32_t original_pcm_bytes_net;// network byte order
    uint64_t audio_capture_ts_ns_net; // network byte order
    uint32_t video_capture_ts_ms_net; // network byte order
};

struct AudioFormatHeader {
    uint32_t sample_rate_net;     // Hz
    uint16_t channels_net;        // 1=mono,2=stereo
    uint16_t bits_per_sample_net; // e.g. 16
    uint16_t stream_id_net;       // optional stream discriminator
};
#pragma pack(pop)

// ===== Runtime state =====
struct ReceivedDatagram {
    std::vector<uint8_t> bytes;
    size_t length = 0;
    std::chrono::steady_clock::time_point recvTime;
};

struct DecodedPcmBlock {
    uint32_t block_id = 0;
    uint64_t audio_capture_ts_ns = 0;
    uint32_t video_capture_ts_ms = 0;
    std::vector<uint8_t> headerBytes; // raw bytes from the datagram header portion
    std::vector<uint8_t> pcm;         // decoded PCM
};

struct BlockAssemblyState {
    uint16_t k = 0;
    uint16_t m = 0;
    uint32_t original_pcm_bytes = 0;
    uint64_t audio_capture_ts_ns = 0;
    uint32_t video_capture_ts_ms = 0;
    std::map<uint32_t, std::vector<uint8_t>> receivedShards;
    std::chrono::steady_clock::time_point firstSeen;
    std::vector<uint8_t> headerBytes;
};

struct AudioFormat {
    uint32_t sampleRate = 48000;
    uint16_t channels = 2;
    uint16_t bitsPerSample = 16;
    uint16_t streamId = 0;
};

// Threading primitives
std::atomic<bool> g_audioRunning{false};
std::atomic<bool> g_audioPlaybackReady{false};
std::atomic<uint16_t> g_activeStreamId{0};
std::thread g_udpThread;
std::thread g_fecThread;
std::thread g_playbackThread;

// Queues
moodycamel::ConcurrentQueue<ReceivedDatagram> g_udpQueue;
moodycamel::ConcurrentQueue<DecodedPcmBlock> g_pcmQueue;

// Block assembly state
std::mutex g_blockMutex;
std::unordered_map<uint32_t, BlockAssemblyState> g_blocks;

// Logging cadence
std::mutex g_logMutex;
std::optional<int64_t> g_lastLoggedSecond; // server seconds

// Video delay estimation (ns on client clock)
std::mutex g_videoDelayMutex;
double g_videoDelayEstimateNs = 0.0;

// Playback structures
struct AudioPlaybackContext {
    CComPtr<IAudioClient> audioClient;
    CComPtr<IAudioRenderClient> renderClient;
    CComPtr<IAudioClock> audioClock;
    HANDLE hEvent = nullptr;
    WAVEFORMATEX mixFormat{};
    uint32_t bufferFrameCount = 0;
    uint32_t bytesPerFrame = 0;
    std::chrono::steady_clock::time_point playbackStart{};
    uint64_t framesSubmitted = 0;
};

AudioPlaybackContext g_playCtx;
AudioFormat g_currentFormat{};
std::mutex g_playbackMutex;

// ===== Helpers =====
uint16_t FromNet16(uint16_t v) { return ntohs(v); }
uint32_t FromNet32(uint32_t v) { return ntohl(v); }
uint64_t FromNet64(uint64_t v) {
    uint32_t high = static_cast<uint32_t>(v >> 32);
    uint32_t low = static_cast<uint32_t>(v & 0xFFFFFFFFu);
    uint64_t host = (static_cast<uint64_t>(ntohl(low)) << 32) | ntohl(high);
    return host;
}

std::string ToHex(const std::vector<uint8_t>& data) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (uint8_t b : data) {
        oss << std::setw(2) << static_cast<int>(b);
    }
    return oss.str();
}

std::wstring ToWide(const std::string& str) {
    if (str.empty()) return L"";
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), nullptr, 0);
    if (size_needed <= 0) return L"";
    std::wstring wstr(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), wstr.data(), size_needed);
    return wstr;
}

void LogHeaderAndPcm(const DecodedPcmBlock& blk) {
    const int64_t sec = static_cast<int64_t>(blk.audio_capture_ts_ns / 1'000'000'000ULL);
    std::lock_guard<std::mutex> lk(g_logMutex);
    if (g_lastLoggedSecond.has_value() && g_lastLoggedSecond.value() == sec) {
        return;
    }
    g_lastLoggedSecond = sec;
    std::vector<uint8_t> combined;
    combined.reserve(blk.headerBytes.size() + blk.pcm.size());
    combined.insert(combined.end(), blk.headerBytes.begin(), blk.headerBytes.end());
    combined.insert(combined.end(), blk.pcm.begin(), blk.pcm.end());
    DebugLog(L"[AudioDump] second=" + std::to_wstring(sec) + L" block=" + std::to_wstring(blk.block_id) +
             L" bytes=" + std::to_wstring(combined.size()) +
             L" hex=" + ToWide(ToHex(combined)));
}

bool ParseHeader(const ReceivedDatagram& dgram, AudioUdpHeader& outHeader) {
    if (dgram.length < sizeof(AudioUdpHeader)) return false;
    std::memcpy(&outHeader, dgram.bytes.data(), sizeof(AudioUdpHeader));
    if (outHeader.header_bytes < sizeof(AudioUdpHeader) || outHeader.header_bytes > dgram.length) {
        return false;
    }
    return true;
}

bool ParseShard(const ReceivedDatagram& dgram, const AudioUdpHeader& hdr, DecodedPcmBlock& decodedOut) {
    const size_t payloadOffset = hdr.header_bytes;
    if (dgram.length < payloadOffset + sizeof(AudioShardHeader)) return false;

    AudioShardHeader shardHdr{};
    std::memcpy(&shardHdr, dgram.bytes.data() + payloadOffset, sizeof(AudioShardHeader));
    const uint32_t block_id = FromNet32(shardHdr.block_id_net);
    const uint16_t shard_index = FromNet16(shardHdr.shard_index_net);
    const uint16_t k = FromNet16(shardHdr.k_net);
    const uint16_t m = FromNet16(shardHdr.m_net);
    const uint32_t original_pcm_bytes = FromNet32(shardHdr.original_pcm_bytes_net);
    const uint64_t audio_ts_ns = FromNet64(shardHdr.audio_capture_ts_ns_net);
    const uint32_t video_ts_ms = FromNet32(shardHdr.video_capture_ts_ms_net);

    const uint8_t* payloadPtr = dgram.bytes.data() + hdr.header_bytes + sizeof(AudioShardHeader);
    const size_t payloadLen = dgram.length - hdr.header_bytes - sizeof(AudioShardHeader);

    // Add to block state
    {
        std::lock_guard<std::mutex> lk(g_blockMutex);
        auto& blk = g_blocks[block_id];
        if (blk.receivedShards.empty()) {
            blk.k = k;
            blk.m = m;
            blk.original_pcm_bytes = original_pcm_bytes;
            blk.audio_capture_ts_ns = audio_ts_ns;
            blk.video_capture_ts_ms = video_ts_ms;
            blk.firstSeen = dgram.recvTime;
            blk.headerBytes.assign(dgram.bytes.begin(), dgram.bytes.begin() + hdr.header_bytes);
        }

        // timeout purge
        const auto now = dgram.recvTime;
        for (auto it = g_blocks.begin(); it != g_blocks.end();) {
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second.firstSeen).count() > AUDIO_ASSEMBLY_TIMEOUT_MS) {
                it = g_blocks.erase(it);
            } else {
                ++it;
            }
        }

        if (payloadLen == 0) return false;
        blk.receivedShards[shard_index] = std::vector<uint8_t>(payloadPtr, payloadPtr + payloadLen);

        if (blk.receivedShards.size() >= blk.k && blk.k > 0) {
            std::vector<uint8_t> decoded;
            if (DecodeFEC_ISAL(blk.receivedShards, blk.k, blk.m, blk.original_pcm_bytes, decoded)) {
                decodedOut.block_id = block_id;
                decodedOut.audio_capture_ts_ns = blk.audio_capture_ts_ns;
                decodedOut.video_capture_ts_ms = blk.video_capture_ts_ms;
                decodedOut.headerBytes = blk.headerBytes;
                decodedOut.pcm = std::move(decoded);
                g_blocks.erase(block_id);
                return true;
            } else {
                g_blocks.erase(block_id);
            }
        }
    }
    return false;
}

bool ParseFormat(const ReceivedDatagram& dgram, const AudioUdpHeader& hdr, AudioFormat& outFmt) {
    const size_t payloadOffset = hdr.header_bytes;
    if (dgram.length < payloadOffset + sizeof(AudioFormatHeader)) return false;
    AudioFormatHeader fmt{};
    std::memcpy(&fmt, dgram.bytes.data() + payloadOffset, sizeof(AudioFormatHeader));
    outFmt.sampleRate = FromNet32(fmt.sample_rate_net);
    outFmt.channels = FromNet16(fmt.channels_net);
    outFmt.bitsPerSample = FromNet16(fmt.bits_per_sample_net);
    outFmt.streamId = FromNet16(fmt.stream_id_net);
    return true;
}

void UdpReceiveLoop() {
    SOCKET sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock == INVALID_SOCKET) {
        DebugLog(L"AudioUdpReceiveThread: socket creation failed");
        return;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(RECEIVE_PORT_AUDIO);
    inet_pton(AF_INET, RECEIVE_IP_AUDIO, &addr.sin_addr);

    if (bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        DebugLog(L"AudioUdpReceiveThread: bind failed");
        closesocket(sock);
        return;
    }

    while (g_audioRunning.load(std::memory_order_acquire)) {
        ReceivedDatagram d{};
        d.bytes.resize(65000);
        int recvLen = recvfrom(sock, reinterpret_cast<char*>(d.bytes.data()), static_cast<int>(d.bytes.size()), 0, nullptr, nullptr);
        if (recvLen <= 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        d.length = static_cast<size_t>(recvLen);
        d.bytes.resize(d.length);
        d.recvTime = std::chrono::steady_clock::now();
        g_udpQueue.enqueue(std::move(d));
    }

    closesocket(sock);
}

void HandlePcmBlock(DecodedPcmBlock&& blk) {
    LogHeaderAndPcm(blk);
    g_pcmQueue.enqueue(std::move(blk));
}

void FecAssembleLoop() {
    while (g_audioRunning.load(std::memory_order_acquire)) {
        ReceivedDatagram dgram;
        if (!g_udpQueue.try_dequeue(dgram)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        AudioUdpHeader hdr{};
        if (!ParseHeader(dgram, hdr)) {
            continue;
        }

        if (hdr.payload_type == static_cast<uint8_t>(AudioPayloadType::AUDIO_SHARD)) {
            DecodedPcmBlock blk;
            if (ParseShard(dgram, hdr, blk)) {
                HandlePcmBlock(std::move(blk));
            }
        } else if (hdr.payload_type == static_cast<uint8_t>(AudioPayloadType::AUDIO_FORMAT)) {
            AudioFormat fmt{};
            if (ParseFormat(dgram, hdr, fmt)) {
                std::lock_guard<std::mutex> lk(g_playbackMutex);
                g_currentFormat = fmt;
                g_activeStreamId.store(fmt.streamId, std::memory_order_release);
                g_audioPlaybackReady.store(false, std::memory_order_release);
                DebugLog(L"Audio format update: "
                    + std::to_wstring(fmt.sampleRate) + L" Hz, "
                    + std::to_wstring(fmt.channels) + L" ch, "
                    + std::to_wstring(fmt.bitsPerSample) + L" bps, stream="
                    + std::to_wstring(fmt.streamId));
            }
        }
    }
}

bool InitializePlaybackContext(AudioPlaybackContext& ctx, const AudioFormat& fmt) {
    ctx = AudioPlaybackContext{};

    CComPtr<IMMDeviceEnumerator> enumerator;
    HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, IID_PPV_ARGS(&enumerator));
    if (FAILED(hr)) {
        DebugLog(L"AudioPlayback: MMDeviceEnumerator creation failed");
        return false;
    }

    CComPtr<IMMDevice> device;
    hr = enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device);
    if (FAILED(hr)) {
        DebugLog(L"AudioPlayback: GetDefaultAudioEndpoint failed");
        return false;
    }

    hr = device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, reinterpret_cast<void**>(&ctx.audioClient));
    if (FAILED(hr)) {
        DebugLog(L"AudioPlayback: Activate IAudioClient failed");
        return false;
    }

    WAVEFORMATEX wfx{};
    wfx.wFormatTag = WAVE_FORMAT_PCM;
    wfx.nSamplesPerSec = fmt.sampleRate;
    wfx.nChannels = fmt.channels;
    wfx.wBitsPerSample = fmt.bitsPerSample;
    wfx.nBlockAlign = (wfx.nChannels * wfx.wBitsPerSample) / 8;
    wfx.nAvgBytesPerSec = wfx.nBlockAlign * wfx.nSamplesPerSec;

    WAVEFORMATEX* closest = nullptr;
    hr = ctx.audioClient->IsFormatSupported(AUDCLNT_SHAREMODE_SHARED, &wfx, &closest);
    if (hr == S_FALSE && closest) {
        DebugLog(L"AudioPlayback: Format not supported, using mix format.");
        std::memcpy(&wfx, closest, sizeof(WAVEFORMATEX));
        CoTaskMemFree(closest);
    } else if (FAILED(hr)) {
        DebugLog(L"AudioPlayback: IsFormatSupported failed, using mix format fallback.");
        hr = ctx.audioClient->GetMixFormat(&closest);
        if (FAILED(hr) || !closest) {
            return false;
        }
        std::memcpy(&wfx, closest, sizeof(WAVEFORMATEX));
        CoTaskMemFree(closest);
    }
    ctx.mixFormat = wfx;
    ctx.bytesPerFrame = wfx.nBlockAlign;

    const REFERENCE_TIME bufferDuration = 100 * 10'000; // 100ms
    hr = ctx.audioClient->Initialize(AUDCLNT_SHAREMODE_SHARED,
                                     AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                                     bufferDuration,
                                     0,
                                     &wfx,
                                     nullptr);
    if (FAILED(hr)) {
        DebugLog(L"AudioPlayback: Initialize failed");
        return false;
    }

    hr = ctx.audioClient->GetService(IID_PPV_ARGS(&ctx.renderClient));
    if (FAILED(hr)) {
        DebugLog(L"AudioPlayback: GetService(render) failed");
        return false;
    }
    hr = ctx.audioClient->GetService(IID_PPV_ARGS(&ctx.audioClock));
    if (FAILED(hr)) {
        DebugLog(L"AudioPlayback: GetService(clock) failed");
        return false;
    }

    hr = ctx.audioClient->GetBufferSize(&ctx.bufferFrameCount);
    if (FAILED(hr)) {
        DebugLog(L"AudioPlayback: GetBufferSize failed");
        return false;
    }

    ctx.hEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!ctx.hEvent) {
        DebugLog(L"AudioPlayback: CreateEvent failed");
        return false;
    }
    hr = ctx.audioClient->SetEventHandle(ctx.hEvent);
    if (FAILED(hr)) {
        DebugLog(L"AudioPlayback: SetEventHandle failed");
        CloseHandle(ctx.hEvent);
        ctx.hEvent = nullptr;
        return false;
    }

    hr = ctx.audioClient->Start();
    if (FAILED(hr)) {
        DebugLog(L"AudioPlayback: Start failed");
        CloseHandle(ctx.hEvent);
        ctx.hEvent = nullptr;
        return false;
    }

    ctx.playbackStart = std::chrono::steady_clock::now();
    ctx.framesSubmitted = 0;
    return true;
}

uint64_t ComputeQueuedDurationMs(const std::deque<DecodedPcmBlock>& queue, const AudioPlaybackContext& ctx) {
    uint64_t bytes = 0;
    for (const auto& blk : queue) bytes += blk.pcm.size();
    if (ctx.bytesPerFrame == 0 || ctx.mixFormat.nSamplesPerSec == 0) return 0;
    const uint64_t frames = bytes / ctx.bytesPerFrame;
    return (frames * 1000) / ctx.mixFormat.nSamplesPerSec;
}

void AdjustForSync(std::vector<uint8_t>& pcm, const DecodedPcmBlock& blk, AudioPlaybackContext& ctx) {
    if (ctx.mixFormat.nSamplesPerSec == 0 || ctx.bytesPerFrame == 0) return;
    const uint64_t audio_pts_client_ns = blk.audio_capture_ts_ns - g_TimeOffsetNs.load(std::memory_order_acquire);
    double video_delay_ns = 0.0;
    {
        std::lock_guard<std::mutex> lk(g_videoDelayMutex);
        video_delay_ns = g_videoDelayEstimateNs;
    }
    const uint64_t target_ns = static_cast<uint64_t>(static_cast<double>(audio_pts_client_ns) + video_delay_ns);

    const auto now = std::chrono::steady_clock::now();
    const uint64_t elapsed_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - ctx.playbackStart).count());
    const uint64_t expected_play_ns = elapsed_ns;

    int64_t sync_error_ns = static_cast<int64_t>(expected_play_ns) - static_cast<int64_t>(target_ns);
    const int64_t tolerance_ns = AUDIO_SYNC_TOLERANCE_MS * 1'000'000LL;
    if (std::abs(sync_error_ns) <= tolerance_ns) return;

    const uint32_t framesPer5ms = static_cast<uint32_t>((ctx.mixFormat.nSamplesPerSec * 5) / 1000);
    const uint32_t bytesPer5ms = framesPer5ms * ctx.bytesPerFrame;
    if (bytesPer5ms == 0 || bytesPer5ms > pcm.size()) return;

    if (sync_error_ns > tolerance_ns) {
        pcm.erase(pcm.begin(), pcm.begin() + bytesPer5ms);
    } else if (sync_error_ns < -tolerance_ns) {
        pcm.insert(pcm.begin(), bytesPer5ms, 0);
    }
}

void AudioPlaybackLoop() {
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    std::deque<DecodedPcmBlock> localQueue;
    AudioFormat activeFmt = g_currentFormat;

    if (!InitializePlaybackContext(g_playCtx, activeFmt)) {
        DebugLog(L"AudioPlayback: failed to init playback context");
        return;
    }
    g_audioPlaybackReady.store(true, std::memory_order_release);

    while (g_audioRunning.load(std::memory_order_acquire)) {
        DecodedPcmBlock blk;
        while (g_pcmQueue.try_dequeue(blk)) {
            if (blk.headerBytes.empty() && blk.pcm.empty()) continue;
            if (blk.pcm.empty()) continue;
            if (g_activeStreamId.load(std::memory_order_acquire) != 0 &&
                blk.headerBytes.size() >= sizeof(AudioUdpHeader)) {
                // Accept all stream IDs for now; future stream filtering can be added here.
            }
            localQueue.push_back(std::move(blk));
        }

        // Handle format changes
        {
            std::lock_guard<std::mutex> lk(g_playbackMutex);
            if (g_currentFormat.streamId != activeFmt.streamId ||
                g_currentFormat.sampleRate != activeFmt.sampleRate ||
                g_currentFormat.channels != activeFmt.channels ||
                g_currentFormat.bitsPerSample != activeFmt.bitsPerSample) {
                DebugLog(L"AudioPlayback: reinitializing for new format.");
                if (g_playCtx.audioClient) {
                    g_playCtx.audioClient->Stop();
                }
                if (g_playCtx.hEvent) {
                    CloseHandle(g_playCtx.hEvent);
                    g_playCtx.hEvent = nullptr;
                }
                activeFmt = g_currentFormat;
                if (!InitializePlaybackContext(g_playCtx, activeFmt)) {
                    DebugLog(L"AudioPlayback: reinit failed");
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }
                localQueue.clear();
                g_audioPlaybackReady.store(true, std::memory_order_release);
            }
        }

        // Jitter buffer: wait until we have enough data
        const uint64_t queuedMs = ComputeQueuedDurationMs(localQueue, g_playCtx);
        if (!localQueue.empty() && queuedMs < AUDIO_JITTER_TARGET_MS) {
            WaitForSingleObject(g_playCtx.hEvent, 5);
            continue;
        }

        if (localQueue.empty()) {
            WaitForSingleObject(g_playCtx.hEvent, 5);
            continue;
        }

        UINT32 padding = 0;
        if (FAILED(g_playCtx.audioClient->GetCurrentPadding(&padding))) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        const UINT32 framesAvailable = g_playCtx.bufferFrameCount > padding
            ? (g_playCtx.bufferFrameCount - padding)
            : 0;
        if (framesAvailable == 0) {
            WaitForSingleObject(g_playCtx.hEvent, 2);
            continue;
        }

        DecodedPcmBlock current = std::move(localQueue.front());
        localQueue.pop_front();
        AdjustForSync(current.pcm, current, g_playCtx);

        size_t bytesToWrite = std::min<size_t>(current.pcm.size(), static_cast<size_t>(framesAvailable) * g_playCtx.bytesPerFrame);
        if (bytesToWrite == 0) {
            continue;
        }

        BYTE* buffer = nullptr;
        if (FAILED(g_playCtx.renderClient->GetBuffer(static_cast<UINT32>(bytesToWrite / g_playCtx.bytesPerFrame), &buffer))) {
            continue;
        }

        std::memcpy(buffer, current.pcm.data(), bytesToWrite);
        g_playCtx.renderClient->ReleaseBuffer(static_cast<UINT32>(bytesToWrite / g_playCtx.bytesPerFrame), 0);
        g_playCtx.framesSubmitted += bytesToWrite / g_playCtx.bytesPerFrame;

        // Push leftover back to front
        if (bytesToWrite < current.pcm.size()) {
            current.pcm.erase(current.pcm.begin(), current.pcm.begin() + bytesToWrite);
            localQueue.push_front(std::move(current));
        }
    }

    if (g_playCtx.audioClient) {
        g_playCtx.audioClient->Stop();
    }
    if (g_playCtx.hEvent) {
        CloseHandle(g_playCtx.hEvent);
        g_playCtx.hEvent = nullptr;
    }
    CoUninitialize();
}

} // namespace

void StartAudioPipeline() {
    bool expected = false;
    if (!g_audioRunning.compare_exchange_strong(expected, true)) {
        return;
    }
    g_udpThread = std::thread(UdpReceiveLoop);
    g_fecThread = std::thread(FecAssembleLoop);
    g_playbackThread = std::thread(AudioPlaybackLoop);
}

void ShutdownAudioPipeline() {
    bool expected = true;
    if (!g_audioRunning.compare_exchange_strong(expected, false)) {
        return;
    }
    if (g_udpThread.joinable()) g_udpThread.join();
    if (g_fecThread.joinable()) g_fecThread.join();
    if (g_playbackThread.joinable()) g_playbackThread.join();
}

void NotifyVideoPresent(uint64_t video_capture_ts_ms, uint64_t client_present_ns) {
    // Estimate video path delay: client_present_ns - (video_capture_ts_ms - g_TimeOffsetNs)
    const int64_t server_capture_ns = static_cast<int64_t>(video_capture_ts_ms) * 1'000'000LL;
    const int64_t client_capture_ns = server_capture_ns - g_TimeOffsetNs.load(std::memory_order_acquire);
    const int64_t delay_ns = static_cast<int64_t>(client_present_ns) - client_capture_ns;

    std::lock_guard<std::mutex> lk(g_videoDelayMutex);
    // Simple moving average to smooth fluctuations
    constexpr double alpha = 0.05;
    if (g_videoDelayEstimateNs == 0.0) {
        g_videoDelayEstimateNs = static_cast<double>(delay_ns);
    } else {
        g_videoDelayEstimateNs = (1.0 - alpha) * g_videoDelayEstimateNs + alpha * static_cast<double>(delay_ns);
    }
}
