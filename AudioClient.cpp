#include "AudioClient.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef WINVER
#define WINVER 0x0A00
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <audiopolicy.h>
#include <mmsystem.h>
#include <mmreg.h>
#include <ksmedia.h>
#include <avrt.h>
#include <atlbase.h>
#include <thread>
#include <vector>
#include <map>
#include <chrono>
#include <mutex>
#include <atomic>
#include <numeric>
#include <deque>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <string>
#include <optional>
#include <unordered_map>

#include "AudioPacket.h"
#include "Globals.h"
#include "DebugLog.h"
#include "ReedSolomon.h"
#include "TimeSyncClient.h"
#include "concurrentqueue/concurrentqueue.h"

namespace {

// A/V Sync
std::atomic<uint64_t> g_lastVideoTimestamp = 0;
class MovingAverage {
public:
    MovingAverage(size_t size) : size_(size), sum_(0) {}
    void add(int64_t value) {
        if (values_.size() == size_) {
            sum_ -= values_.front();
            values_.pop_front();
        }
        values_.push_back(value);
        sum_ += value;
    }
    int64_t get_average() const {
        if (values_.empty()) return 0;
        return sum_ / values_.size();
    }
private:
    std::deque<int64_t> values_;
    size_t size_;
    int64_t sum_;
};
MovingAverage g_videoPathDelay(10);

// Thread control
std::atomic<bool> g_audioThreadsRunning = false;
std::thread g_audioUdpReceiveThread;
std::thread g_audioFecAssembleThread;
std::thread g_audioPlaybackThread;

// Data queues
struct ReceivedPacket {
    std::vector<uint8_t> data;
    std::chrono::steady_clock::time_point received_time;
};
moodycamel::ConcurrentQueue<ReceivedPacket> g_receivedPacketQueue;

struct DecodedPcmBlock {
    uint64_t audio_capture_ts_ns;
    uint64_t video_capture_ts_ms;
    std::vector<uint8_t> pcm_data;
    // For logging
    std::vector<uint8_t> header_for_log;
};
moodycamel::ConcurrentQueue<DecodedPcmBlock> g_decodedPcmQueue;
moodycamel::ConcurrentQueue<AudioFormatPayload> g_audioFormatQueue;

void AudioUdpReceiveThread() {
    DebugLog(L"[AudioUDP] Receive thread started.");

    SOCKET sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock == INVALID_SOCKET) {
        DebugLog(L"[AudioUDP] Failed to create socket: " + std::to_wstring(WSAGetLastError()));
        return;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(RECEIVE_PORT_AUDIO);
    inet_pton(AF_INET, RECEIVE_IP_AUDIO, &addr.sin_addr);

    if (bind(sock, (SOCKADDR*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        DebugLog(L"[AudioUDP] Failed to bind socket: " + std::to_wstring(WSAGetLastError()));
        closesocket(sock);
        return;
    }

    // Set a timeout so the loop can exit gracefully
    DWORD timeout = 100; // ms
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));

    while (g_audioThreadsRunning) {
        ReceivedPacket packet;
        packet.data.resize(2048); // Max expected packet size
        sockaddr_in sender_addr{};
        int sender_addr_len = sizeof(sender_addr);

        int recv_len = recvfrom(sock, (char*)packet.data.data(), (int)packet.data.size(), 0, (SOCKADDR*)&sender_addr, &sender_addr_len);

        if (recv_len > 0) {
            packet.data.resize(recv_len);
            packet.received_time = std::chrono::steady_clock::now();
            g_receivedPacketQueue.enqueue(std::move(packet));
        } else if (recv_len == SOCKET_ERROR) {
            int error = WSAGetLastError();
            if (error != WSAETIMEDOUT) {
                DebugLog(L"[AudioUDP] recvfrom failed with error: " + std::to_wstring(error));
                break;
            }
        }
    }

    closesocket(sock);
    DebugLog(L"[AudioUDP] Receive thread stopped.");
}

// Helper for network to host byte order conversion of 64-bit integers
uint64_t Ntohll(uint64_t v) {
    return (static_cast<uint64_t>(ntohl(static_cast<uint32_t>(v >> 32))) |
            (static_cast<uint64_t>(ntohl(static_cast<uint32_t>(v & 0xFFFFFFFFULL))) << 32));
}

struct HostAudioFormat {
    uint32_t sample_rate = 48000;
    uint16_t channels = 2;
    uint16_t bits_per_sample = 16;
    uint16_t bytes_per_frame = 4;   // channels * (bits/8)
    uint32_t channel_mask = 0;
    uint32_t sample_format = 2;     // 1=float32, 2=pcm(int)
};

static uint32_t DefaultChannelMask(uint16_t ch) {
    switch (ch) {
    case 1: return SPEAKER_FRONT_CENTER;
    case 2: return SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT;
    case 4: return SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT;
    case 6: return SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_FRONT_CENTER | SPEAKER_LOW_FREQUENCY |
                   SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT;
    case 8: return SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_FRONT_CENTER | SPEAKER_LOW_FREQUENCY |
                   SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT | SPEAKER_SIDE_LEFT | SPEAKER_SIDE_RIGHT;
    default: return 0;
    }
}

static HostAudioFormat ParseAudioFormatPayload(const AudioFormatPayload& p) {
    HostAudioFormat f{};
    f.sample_rate     = ntohl(p.sample_rate_be);
    f.channels        = ntohs(p.channels_be);
    f.bits_per_sample = ntohs(p.bits_per_sample_be);
    f.bytes_per_frame = ntohs(p.bytes_per_frame_be);
    f.channel_mask    = ntohl(p.channel_mask_be);
    f.sample_format   = ntohl(p.sample_format_be);

    if (f.bytes_per_frame == 0 && f.channels != 0 && f.bits_per_sample != 0) {
        f.bytes_per_frame = static_cast<uint16_t>((f.channels * f.bits_per_sample) / 8);
    }
    if (f.channel_mask == 0) {
        f.channel_mask = DefaultChannelMask(f.channels);
    }
    return f;
}

static bool SameAudioFormat(const HostAudioFormat& a, const HostAudioFormat& b) {
    return a.sample_rate == b.sample_rate &&
           a.channels == b.channels &&
           a.bits_per_sample == b.bits_per_sample &&
           a.bytes_per_frame == b.bytes_per_frame &&
           a.channel_mask == b.channel_mask &&
           a.sample_format == b.sample_format;
}

static WAVEFORMATEXTENSIBLE BuildWaveFormatExtensible(const HostAudioFormat& f) {
    WAVEFORMATEXTENSIBLE wfx{};
    wfx.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
    wfx.Format.nChannels = f.channels;
    wfx.Format.nSamplesPerSec = f.sample_rate;
    wfx.Format.wBitsPerSample = f.bits_per_sample;
    wfx.Format.nBlockAlign = f.bytes_per_frame;
    wfx.Format.nAvgBytesPerSec = wfx.Format.nSamplesPerSec * wfx.Format.nBlockAlign;
    wfx.Format.cbSize = sizeof(WAVEFORMATEXTENSIBLE) - sizeof(WAVEFORMATEX);
    wfx.Samples.wValidBitsPerSample = f.bits_per_sample;
    wfx.dwChannelMask = f.channel_mask;
    wfx.SubFormat = (f.sample_format == 1) ? KSDATAFORMAT_SUBTYPE_IEEE_FLOAT : KSDATAFORMAT_SUBTYPE_PCM;
    return wfx;
}

struct BlockState {
    uint32_t block_id;
    uint8_t k, m;
    uint32_t original_pcm_bytes;
    uint64_t audio_capture_ts_ns;
    uint64_t video_capture_ts_ms;
    std::chrono::steady_clock::time_point first_shard_received_time;
    std::map<uint32_t, std::vector<uint8_t>> shards;
    std::vector<uint8_t> first_header; // For logging
};
std::map<uint32_t, BlockState> g_blockAssemblyBuffer;
std::mutex g_blockAssemblyMutex;

void AudioFecAssembleThread() {
    DebugLog(L"[AudioFEC] Assemble thread started.");

    auto last_timeout_check = std::chrono::steady_clock::now();

    while (g_audioThreadsRunning) {
        ReceivedPacket packet;
        bool has_packet = g_receivedPacketQueue.try_dequeue(packet);

        if (has_packet) {
            if (packet.data.size() < sizeof(AudioUdpHeader)) {
                DebugLog(L"[AudioFEC] Received packet too small for header.");
                continue;
            }

            AudioUdpHeader header;
            memcpy(&header, packet.data.data(), sizeof(AudioUdpHeader));

            if (ntohl(header.magic_be) != 0x52415544) { // 'RAUD'
                DebugLog(L"[AudioFEC] Invalid audio packet magic.");
                continue;
            }

            if (header.packet_type == AUDIO_PACKET_TYPE_SHARD) {
                uint32_t block_id = ntohl(header.block_id_be);
                uint16_t shard_index = ntohs(header.shard_index_be);
                bool ready_for_decode = false;

                // Lock scope for modifying the assembly buffer
                {
                    std::lock_guard<std::mutex> lock(g_blockAssemblyMutex);
                    auto& block_state = g_blockAssemblyBuffer[block_id];

                    if (block_state.shards.empty()) { // First shard for this block
                        block_state.block_id = block_id;
                        block_state.k = header.k;
                        block_state.m = header.m;
                        block_state.original_pcm_bytes = ntohl(header.original_pcm_bytes_be);
                        block_state.audio_capture_ts_ns = Ntohll(header.audio_capture_ts_ns_be);
                        block_state.video_capture_ts_ms = Ntohll(header.video_capture_ts_ms_be);
                        block_state.first_shard_received_time = packet.received_time;
                        block_state.first_header.assign(packet.data.begin(), packet.data.begin() + sizeof(AudioUdpHeader));
                    }

                    const uint8_t* payload_ptr = packet.data.data() + sizeof(AudioUdpHeader);
                    size_t payload_len = packet.data.size() - sizeof(AudioUdpHeader);
                    block_state.shards[shard_index].assign(payload_ptr, payload_ptr + payload_len);

                    if (block_state.k > 0 && block_state.shards.size() >= block_state.k) {
                        ready_for_decode = true;
                    }
                } // End lock

                if (ready_for_decode) {
                    BlockState state_to_decode;
                    bool extracted = false;
                    {
                        std::lock_guard<std::mutex> lock(g_blockAssemblyMutex);
                        auto it = g_blockAssemblyBuffer.find(block_id);
                        if (it != g_blockAssemblyBuffer.end() && it->second.shards.size() >= it->second.k) {
                            state_to_decode = std::move(it->second);
                            g_blockAssemblyBuffer.erase(it);
                            extracted = true;
                        }
                    }

                    if (extracted) {
                        DecodedPcmBlock pcm_block;
                        pcm_block.audio_capture_ts_ns = state_to_decode.audio_capture_ts_ns;
                        pcm_block.video_capture_ts_ms = state_to_decode.video_capture_ts_ms;
                        pcm_block.header_for_log = state_to_decode.first_header;

                        if (DecodeFEC_ISAL(state_to_decode.shards, state_to_decode.k, state_to_decode.m, state_to_decode.original_pcm_bytes, pcm_block.pcm_data)) {
                            g_decodedPcmQueue.enqueue(std::move(pcm_block));
                        } else {
                            DebugLog(L"[AudioFEC] FEC decoding failed for block " + std::to_wstring(block_id));
                        }
                    }
                }
            } else if (header.packet_type == AUDIO_PACKET_TYPE_FORMAT) {
                if (packet.data.size() >= sizeof(AudioUdpHeader) + sizeof(AudioFormatPayload)) {
                    AudioFormatPayload format_payload;
                    memcpy(&format_payload, packet.data.data() + sizeof(AudioUdpHeader), sizeof(AudioFormatPayload));
                    g_audioFormatQueue.enqueue(format_payload);
                }
            }
        } else {
             std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Periodically check for timeouts
        auto now = std::chrono::steady_clock::now();
        if (now - last_timeout_check > std::chrono::milliseconds(20)) { // Check more frequently
            std::lock_guard<std::mutex> lock(g_blockAssemblyMutex);
            for (auto it = g_blockAssemblyBuffer.begin(); it != g_blockAssemblyBuffer.end();) {
                if (std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second.first_shard_received_time).count() > AUDIO_ASSEMBLY_TIMEOUT_MS) {
                    DebugLog(L"[AudioFEC] Block " + std::to_wstring(it->first) + L" timed out.");
                    it = g_blockAssemblyBuffer.erase(it);
                } else {
                    ++it;
                }
            }
            last_timeout_check = now;
        }
    }
    DebugLog(L"[AudioFEC] Assemble thread stopped.");
}

void HexDumpBytes(const std::vector<uint8_t>& header, const std::vector<uint8_t>& pcm) {
    if (header.empty() && pcm.empty()) return;

    std::vector<uint8_t> data = header;
    data.insert(data.end(), pcm.begin(), pcm.end());

    constexpr char kHex[] = "0123456789ABCDEF";
    std::string hex;
    hex.reserve(data.size() * 2);
    for (uint8_t b : data) {
        hex.push_back(kHex[(b >> 4) & 0xF]);
        hex.push_back(kHex[b & 0xF]);
    }
    // Using custom ConvertToWString function if available, otherwise std::wstring constructor
    std::wstring wHex(hex.begin(), hex.end());
    DebugLog(L"[AudioDump] " + wHex);
}

void AudioPlaybackThread() {
    DebugLog(L"[AudioPlayback] Playback thread started.");

    if (FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED))) {
        DebugLog(L"[AudioPlayback] CoInitializeEx failed.");
        return;
    }

    CComPtr<IMMDeviceEnumerator> enumerator;
    HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, IID_PPV_ARGS(&enumerator));
    if (FAILED(hr)) {
        DebugLog(L"[AudioPlayback] Failed to create device enumerator.");
        return;
    }

    std::optional<HostAudioFormat> activeFormat; // サーバPCMのフォーマット（重複formatは無視）

    while (g_audioThreadsRunning) {
        CComPtr<IAudioClient> audioClient;
        CComPtr<IAudioRenderClient> renderClient;
        UINT32 bufferFrameCount;

        // サーバは1秒に1回 format packet を送る設計なので、ここではキューをDrainして最新のみ採用。
        AudioFormatPayload fmtNet{};
        HostAudioFormat latest{};
        bool gotFmt = false;
        while (g_audioFormatQueue.try_dequeue(fmtNet)) {
            latest = ParseAudioFormatPayload(fmtNet);
            gotFmt = true;
        }
        if (gotFmt) {
            if (!activeFormat || !SameAudioFormat(*activeFormat, latest)) {
                activeFormat = latest;
            }
        }
        if (!activeFormat) {
            activeFormat = HostAudioFormat{}; // 初回formatがまだ来ない場合の暫定
        }

        WAVEFORMATEXTENSIBLE desiredExt = BuildWaveFormatExtensible(*activeFormat);
        WAVEFORMATEX* desiredWfx = reinterpret_cast<WAVEFORMATEX*>(&desiredExt);

        CComPtr<IMMDevice> device;
        hr = enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device);
        if (FAILED(hr)) {
            DebugLog(L"[AudioPlayback] Failed to get default audio endpoint.");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        hr = device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&audioClient);
        if (FAILED(hr)) {
            DebugLog(L"[AudioPlayback] Failed to activate audio client.");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        WAVEFORMATEX* closestMatch = nullptr;
        hr = audioClient->IsFormatSupported(AUDCLNT_SHAREMODE_SHARED, desiredWfx, &closestMatch);
        if (FAILED(hr)) {
            DebugLog(L"[AudioPlayback] Format not supported.");
            if (closestMatch) CoTaskMemFree(closestMatch);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // S_FALSE のときは closestMatch 推奨（EXTENSIBLE⇔非EXTENSIBLE差分だけのことも多い）
        WAVEFORMATEX* renderWfx = desiredWfx;
        if (hr == S_FALSE && closestMatch) {
            renderWfx = closestMatch;
        }
        const uint16_t renderBlockAlign = renderWfx->nBlockAlign;
        if (renderBlockAlign != activeFormat->bytes_per_frame) {
            DebugLog(L"[AudioPlayback] Render blockAlign != server bytes_per_frame. "
                     L"Need sample conversion (float/int16 etc). For now output silence.");
        }

        hr = audioClient->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, 10000000, 0, renderWfx, nullptr);
        if (FAILED(hr)) {
            DebugLog(L"[AudioPlayback] Failed to initialize audio client.");
            if (closestMatch) CoTaskMemFree(closestMatch);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        if (closestMatch) CoTaskMemFree(closestMatch);

        hr = audioClient->GetBufferSize(&bufferFrameCount);
        if (FAILED(hr)) {
            DebugLog(L"[AudioPlayback] Failed to get buffer size.");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        hr = audioClient->GetService(IID_PPV_ARGS(&renderClient));
        if (FAILED(hr)) {
            DebugLog(L"[AudioPlayback] Failed to get render client.");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        std::deque<DecodedPcmBlock> jitterBuffer;
        size_t pcmReadOffset = 0;
        const size_t targetJitterFrames = (static_cast<size_t>(activeFormat->sample_rate) * AUDIO_JITTER_TARGET_MS) / 1000;
        bool isPlaying = false;

        hr = audioClient->Start();
        if (FAILED(hr)) {
            DebugLog(L"[AudioPlayback] Failed to start audio client.");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        uint64_t lastLoggedSecond = 0;
        auto last_correction_time = std::chrono::steady_clock::now();

        while (g_audioThreadsRunning) {
            // 周期formatは無視。同一フォーマットならスルー、変化があれば再初期化。
            AudioFormatPayload newNet{};
            HostAudioFormat newHost{};
            bool gotNew = false;
            while (g_audioFormatQueue.try_dequeue(newNet)) {
                newHost = ParseAudioFormatPayload(newNet);
                gotNew = true;
            }
            if (gotNew && activeFormat && !SameAudioFormat(*activeFormat, newHost)) {
                DebugLog(L"[AudioPlayback] Audio format changed. Re-initializing.");
                *activeFormat = newHost;
                break;
            }

            DecodedPcmBlock pcm_block;
            while(g_decodedPcmQueue.try_dequeue(pcm_block)) {
                jitterBuffer.push_back(std::move(pcm_block));
            }

            size_t total_buffered_frames = 0;
            for(const auto& block : jitterBuffer) {
                total_buffered_frames += block.pcm_data.size() / activeFormat->bytes_per_frame;
            }

            if (!isPlaying && total_buffered_frames >= targetJitterFrames) {
                DebugLog(L"[AudioPlayback] Jitter buffer filled. Starting playback.");
                isPlaying = true;
            }

            if (isPlaying) {
                UINT32 padding;
                hr = audioClient->GetCurrentPadding(&padding);
                if (FAILED(hr)) break;

                UINT32 framesAvailable = bufferFrameCount - padding;
                if (framesAvailable > 0 && !jitterBuffer.empty()) {
                    DecodedPcmBlock& currentBlock = jitterBuffer.front();

                    // A/V Sync Logic
                    int64_t sync_error_ns = 0;
                    if (g_lastVideoTimestamp > 0) {
                        int64_t video_pts_client_ns = (g_lastVideoTimestamp * 1000000) - g_TimeOffsetNs;
                        int64_t audio_pts_client_ns = currentBlock.audio_capture_ts_ns - g_TimeOffsetNs;

                        int64_t video_path_delay_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - video_pts_client_ns;
                        g_videoPathDelay.add(video_path_delay_ns);

                        int64_t audio_play_target_ns = audio_pts_client_ns + g_videoPathDelay.get_average();

                        int64_t audio_play_head_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count() +
                                                   (padding * 1000000000 / activeFormat->sample_rate);
                        sync_error_ns = audio_play_head_ns - audio_play_target_ns;
                    }

                    UINT32 framesRemainingInBlock = (currentBlock.pcm_data.size() - pcmReadOffset) / activeFormat->bytes_per_frame;
                    UINT32 framesToWrite = std::min(framesAvailable, framesRemainingInBlock);

                    auto now = std::chrono::steady_clock::now();
                    if (abs(sync_error_ns) > AUDIO_SYNC_TOLERANCE_MS * 1000000LL && std::chrono::duration_cast<std::chrono::milliseconds>(now - last_correction_time).count() > 200) {
                        if (sync_error_ns > 0) { // Audio is late
                            UINT32 framesToDrop = std::min(framesToWrite, (UINT32)(activeFormat->sample_rate * 0.005));
                            pcmReadOffset += framesToDrop * activeFormat->bytes_per_frame;
                            framesRemainingInBlock -= framesToDrop;
                            framesToWrite = std::min(framesAvailable, framesRemainingInBlock);
                            last_correction_time = now;
                        } else { // Audio is early
                            UINT32 framesToInsert = std::min(framesAvailable, (UINT32)(desiredFormat.nSamplesPerSec * 0.005));
                            BYTE* pData;
                            if (SUCCEEDED(renderClient->GetBuffer(framesToInsert, &pData))) {
                                memset(pData, 0, framesToInsert * renderBlockAlign);
                                renderClient->ReleaseBuffer(framesToInsert, AUDCLNT_BUFFERFLAGS_SILENT);
                                last_correction_time = now;
                            }
                        }
                    }

                    BYTE* pData;
                    if (framesToWrite > 0) {
                        hr = renderClient->GetBuffer(framesToWrite, &pData);
                        if (SUCCEEDED(hr)) {
                            if (renderBlockAlign == activeFormat->bytes_per_frame) {
                                memcpy(pData, currentBlock.pcm_data.data() + pcmReadOffset, framesToWrite * renderBlockAlign);
                            } else {
                                // TODO: ここで float32<->int16 などのサンプル変換を実装する
                                memset(pData, 0, framesToWrite * renderBlockAlign);
                            }
                            renderClient->ReleaseBuffer(framesToWrite, 0);
                            pcmReadOffset += framesToWrite * activeFormat->bytes_per_frame;
                        }
                    }

                    if (pcmReadOffset >= currentBlock.pcm_data.size()) {
                        uint64_t currentSecond = currentBlock.audio_capture_ts_ns / 1000000000;
                        if (currentSecond != lastLoggedSecond) {
                            HexDumpBytes(currentBlock.header_for_log, currentBlock.pcm_data);
                            lastLoggedSecond = currentSecond;
                        }
                        jitterBuffer.pop_front();
                        pcmReadOffset = 0;
                    }

                } else if (jitterBuffer.empty()) {
                    isPlaying = false;
                    DebugLog(L"[AudioPlayback] Jitter buffer empty. Stopping playback.");
                }
            }

            if (!isPlaying || jitterBuffer.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        if(audioClient) audioClient->Stop();
    }

    CoUninitialize();
    DebugLog(L"[AudioPlayback] Playback thread stopped.");
}

} // namespace

void StartAudioThreads() {
    if (g_audioThreadsRunning.exchange(true)) {
        return; // Already running
    }
    DebugLog(L"[AudioClient] Starting audio threads...");
    g_audioUdpReceiveThread = std::thread(AudioUdpReceiveThread);
    g_audioFecAssembleThread = std::thread(AudioFecAssembleThread);
    g_audioPlaybackThread = std::thread(AudioPlaybackThread);
}

void StopAudioThreads() {
    if (!g_audioThreadsRunning.exchange(false)) {
        return; // Already stopped
    }
    DebugLog(L"[AudioClient] Stopping audio threads...");

    if (g_audioUdpReceiveThread.joinable()) {
        g_audioUdpReceiveThread.join();
    }
    if (g_audioFecAssembleThread.joinable()) {
        g_audioFecAssembleThread.join();
    }
    if (g_audioPlaybackThread.joinable()) {
        g_audioPlaybackThread.join();
    }
}

void UpdateVideoTimestamp(uint64_t timestamp) {
    g_lastVideoTimestamp.store(timestamp, std::memory_order_relaxed);
}
