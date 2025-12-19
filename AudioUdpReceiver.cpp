// Standalone audio UDP receiver for packets produced by AudioUdpSender.
//
// This file is intentionally self contained so it can be copied to a client
// project as-is. It is not part of the server build; compile manually when
// needed (e.g., `cl /std:c++17 /EHsc AudioUdpReceiverSample.cpp ws2_32.lib`).
//
// Packet layout (see SendAudioPacket in AudioUdpSender.cpp):
// [u64 captureTimestampMs][u64 qpcTimestamp][u32 sampleRate][u32 channels]
// [u32 frames][u32 sampleCount][float interleavedSamples...]
//
// Usage example:
//   AudioUdpReceiver receiver(8200);
//   receiver.Start([](const ReceivedAudioPacket& packet) {
//       // process audio samples here
//   });
//   ...
//   receiver.Stop();

#include <winsock2.h>
#include <ws2tcpip.h>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <functional>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>
#include <vector>

#include <Windows.h>
#include <mmsystem.h>
#include <mmreg.h>
#include <ks.h>
#include <ksmedia.h>

#include "AudioUdpReceiver.h"
#include "DebugLog.h"

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "winmm.lib")

struct ReceivedAudioPacket {
    uint64_t captureTimestampMs = 0;  // server-side frame timestamp (ms)
    int64_t qpcTimestamp = 0;         // QueryPerformanceCounter ticks
    uint32_t sampleRate = 0;          // Hz
    uint32_t channels = 0;
    uint32_t frames = 0;              // frames per channel contained in payload
    std::vector<float> interleavedSamples;  // size = frames * channels
};

namespace {

constexpr size_t kMaxUdpPayload = 64 * 1024;  // generous buffer for UDP packets
constexpr uint16_t kAudioListenPort = 8200;
constexpr size_t kAudioHeaderBytes = 32;
constexpr size_t kAudioPayloadBytes = 3840;   // server sends 32-byte header + 3840 bytes audio

uint64_t NowSystemNs() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
}

struct QueuedAudioPacket {
    ReceivedAudioPacket packet;
};

constexpr uint64_t kStreamInactivityNs = 1'000'000'000ull;  // 1 second

bool ParseAudioHeader(const uint8_t* data, size_t size, ReceivedAudioPacket& out) {
    if (size != kAudioHeaderBytes) {
        return false;
    }

    size_t offset = 0;
    auto read_u64 = [&](uint64_t& value) {
        std::memcpy(&value, data + offset, sizeof(uint64_t));
        offset += sizeof(uint64_t);
    };
    auto read_u32 = [&](uint32_t& value) {
        std::memcpy(&value, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
    };

    read_u64(out.captureTimestampMs);
    read_u64(reinterpret_cast<uint64_t&>(out.qpcTimestamp));
    read_u32(out.sampleRate);
    read_u32(out.channels);
    read_u32(out.frames);

    if (out.channels == 0 || out.channels > 8) {
        return false;
    }

    if (out.frames == 0) {
        return false;
    }

    if (out.sampleRate < 8'000 || out.sampleRate > 192'000) {
        return false;
    }

    return true;
}

class WinsockScope {
public:
    bool EnsureInitialized() {
        std::call_once(flag_, [&]() {
            WSADATA wsaData{};
            initialized_ = (WSAStartup(MAKEWORD(2, 2), &wsaData) == 0);
        });
        return initialized_;
    }

    ~WinsockScope() {
        if (initialized_) {
            WSACleanup();
        }
    }

private:
    std::once_flag flag_;
    bool initialized_ = false;
};

}  // namespace

class AudioSyncPlayer {
public:
    AudioSyncPlayer() = default;

    ~AudioSyncPlayer() { Stop(); }

    bool Start() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (running_) {
            return true;
        }
        running_ = true;
        lastPacketTimeNs_.store(0, std::memory_order_relaxed);
        playbackThread_ = std::thread(&AudioSyncPlayer::PlaybackLoop, this);
        return true;
    }

    void Stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!running_) {
                return;
            }
            running_ = false;
        }
        cv_.notify_all();
        if (playbackThread_.joinable()) {
            playbackThread_.join();
        }
        CleanupWaveOut(true);
    }

    void Enqueue(const ReceivedAudioPacket& packet) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_) {
            return;
        }
        queue_.push({packet});
        lastPacketTimeNs_.store(NowSystemNs(), std::memory_order_relaxed);
        cv_.notify_one();
    }

private:
    struct ActiveBuffer {
        WAVEHDR header{};
        std::vector<uint8_t> data;
    };

    void PlaybackLoop() {
        while (true) {
            QueuedAudioPacket item;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait_for(lock, std::chrono::milliseconds(100), [&]() { return !running_ || !queue_.empty(); });
                if (!running_ && queue_.empty()) {
                    break;
                }
                if (queue_.empty()) {
                    if (ShouldResetForInactivity()) {
                        CleanupWaveOutLocked(true);
                    }
                    continue;
                }

                item = queue_.front();
                queue_.pop();
            }

            SubmitPacket(item.packet);
            CleanupCompletedHeaders();
        }
        CleanupCompletedHeaders();
    }

    bool EnsureWaveOut(uint32_t sampleRate, uint32_t channels) {
        std::lock_guard<std::mutex> lock(mutex_);
        return EnsureWaveOutLocked(sampleRate, channels);
    }

    bool EnsureWaveOutLocked(uint32_t sampleRate, uint32_t channels) {
        if (waveOut_) {
            if (currentSampleRate_ == sampleRate && currentChannels_ == channels) {
                return true;
            }
            CleanupWaveOutLocked(false);
        }

        WAVEFORMATEXTENSIBLE format{};
        format.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
        format.Format.nChannels = static_cast<WORD>(channels);
        format.Format.nSamplesPerSec = sampleRate;
        format.Format.wBitsPerSample = 32;
        format.Format.nBlockAlign = format.Format.nChannels * format.Format.wBitsPerSample / 8;
        format.Format.nAvgBytesPerSec = format.Format.nSamplesPerSec * format.Format.nBlockAlign;
        format.Format.cbSize = sizeof(WAVEFORMATEXTENSIBLE) - sizeof(WAVEFORMATEX);
        format.Samples.wValidBitsPerSample = 32;
        format.dwChannelMask = (channels == 1) ? SPEAKER_FRONT_CENTER : (SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT);
        format.SubFormat = KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;

        MMRESULT res = waveOutOpen(&waveOut_, WAVE_MAPPER, reinterpret_cast<WAVEFORMATEX*>(&format), 0, 0, CALLBACK_NULL);
        if (res != MMSYSERR_NOERROR) {
            waveOut_ = nullptr;
            return false;
        }

        currentSampleRate_ = sampleRate;
        currentChannels_ = channels;
        return true;
    }

    void SubmitPacket(const ReceivedAudioPacket& packet) {
        if (!EnsureWaveOut(packet.sampleRate, packet.channels)) {
            return;
        }

        ActiveBuffer buffer{};
        buffer.data.resize(packet.interleavedSamples.size() * sizeof(float));
        std::memcpy(buffer.data.data(), packet.interleavedSamples.data(), buffer.data.size());

        buffer.header.lpData = reinterpret_cast<LPSTR>(buffer.data.data());
        buffer.header.dwBufferLength = static_cast<DWORD>(buffer.data.size());

        std::lock_guard<std::mutex> lock(mutex_);
        if (!waveOut_) {
            return;
        }

        if (waveOutPrepareHeader(waveOut_, &buffer.header, sizeof(WAVEHDR)) != MMSYSERR_NOERROR) {
            return;
        }
        if (waveOutWrite(waveOut_, &buffer.header, sizeof(WAVEHDR)) != MMSYSERR_NOERROR) {
            waveOutUnprepareHeader(waveOut_, &buffer.header, sizeof(WAVEHDR));
            return;
        }

        active_.push_back(std::move(buffer));
    }

    void CleanupCompletedHeaders() {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = active_.begin();
        while (it != active_.end()) {
            if (it->header.dwFlags & WHDR_DONE) {
                waveOutUnprepareHeader(waveOut_, &it->header, sizeof(WAVEHDR));
                it = active_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void CleanupWaveOut(bool reset) {
        std::lock_guard<std::mutex> lock(mutex_);
        CleanupWaveOutLocked(reset);
    }

    void CleanupWaveOutLocked(bool reset) {
        if (waveOut_) {
            if (reset) {
                waveOutReset(waveOut_);
            }
            for (auto& buf : active_) {
                waveOutUnprepareHeader(waveOut_, &buf.header, sizeof(WAVEHDR));
            }
            active_.clear();
            waveOutClose(waveOut_);
            waveOut_ = nullptr;
        }
    }

    bool ShouldResetForInactivity() const {
        const uint64_t lastPacketNs = lastPacketTimeNs_.load(std::memory_order_relaxed);
        if (lastPacketNs == 0) {
            return false;
        }
        const uint64_t nowNs = NowSystemNs();
        return nowNs > lastPacketNs && (nowNs - lastPacketNs) > kStreamInactivityNs;
    }

    std::atomic<bool> running_{false};
    std::thread playbackThread_;
    std::queue<QueuedAudioPacket> queue_;
    std::vector<ActiveBuffer> active_;
    std::mutex mutex_;
    std::condition_variable cv_;

    HWAVEOUT waveOut_ = nullptr;
    uint32_t currentSampleRate_ = 0;
    uint32_t currentChannels_ = 0;
    std::atomic<uint64_t> lastPacketTimeNs_{0};
};

class AudioUdpReceiver {
public:
    using PacketCallback = std::function<void(const ReceivedAudioPacket&)>;

    explicit AudioUdpReceiver(uint16_t listenPort = 8200)
        : listenPort_(listenPort) {}

    ~AudioUdpReceiver() { Stop(); }

    bool Start(PacketCallback callback) {
        if (!callback) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (running_) {
            return true;
        }

        if (!winsock_.EnsureInitialized()) {
            return false;
        }

        socket_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (socket_ == INVALID_SOCKET) {
            return false;
        }

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(listenPort_);
        addr.sin_addr.s_addr = htonl(INADDR_ANY);

        if (bind(socket_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
            closesocket(socket_);
            socket_ = INVALID_SOCKET;
            return false;
        }

        running_ = true;
        callback_ = std::move(callback);
        receiveThread_ = std::thread(&AudioUdpReceiver::ReceiveLoop, this);
        return true;
    }

    bool StartWithSynchronizedPlayback() {
        if (!audioPlayer_.Start()) {
            return false;
        }
        if (!Start([this](const ReceivedAudioPacket& packet) { audioPlayer_.Enqueue(packet); })) {
            audioPlayer_.Stop();
            return false;
        }
        return true;
    }

    void Stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!running_) {
                return;
            }
            running_ = false;
            if (socket_ != INVALID_SOCKET) {
                closesocket(socket_);
                socket_ = INVALID_SOCKET;
            }
        }
        if (receiveThread_.joinable()) {
            receiveThread_.join();
        }
        audioPlayer_.Stop();
        std::lock_guard<std::mutex> lock(mutex_);
        callback_ = nullptr;
    }

    bool IsRunning() const { return running_; }

private:
    void ReceiveLoop() {
        std::vector<uint8_t> buffer(kMaxUdpPayload);
        while (running_) {
            int received = recv(socket_, reinterpret_cast<char*>(buffer.data()), static_cast<int>(buffer.size()), 0);
            if (received <= 0) {
                break;  // socket closed or error; exit loop and let Stop() clean up
            }

            // Some environments have been observed to deliver packets where the payload
            // (audio samples) is entirely zeroed while the 32-byte header looks valid.
            // Skip such packets early to avoid enqueuing silence and polluting the log.
            if (received > static_cast<int>(kAudioHeaderBytes)) {
                const uint8_t* payloadBegin = buffer.data() + kAudioHeaderBytes;
                const uint8_t* payloadEnd = buffer.data() + received;
                if (std::all_of(payloadBegin, payloadEnd, [](uint8_t b) { return b == 0; })) {
                    DebugLog(L"[AudioUdpReceiver] Dropping packet with zeroed audio payload.");
                    continue;
                }
            }

            std::wstringstream logStream;
            logStream << L"[AudioUdpReceiver] Received " << received << L" bytes. ";
            logStream << std::hex << std::setw(2) << std::setfill(L'0');

            if (received > 32) {
                logStream << L"Header: ";
                for (int i = 0; i < 32; ++i) {
                    logStream << std::setw(2) << static_cast<int>(buffer[i]) << (i < 31 ? L" " : L"");
                }
                logStream << L" | Payload: ";
                for (int i = 32; i < received; ++i) {
                    logStream << std::setw(2) << static_cast<int>(buffer[i]) << (i < received - 1 ? L" " : L"");
                }
            } else {
                logStream << L"Data: ";
                for (int i = 0; i < received; ++i) {
                    logStream << std::setw(2) << static_cast<int>(buffer[i]) << (i < received - 1 ? L" " : L"");
                }
            }
            DebugLog(logStream.str());

            constexpr size_t kTotalPacketSize = kAudioHeaderBytes + kAudioPayloadBytes;

            if (static_cast<size_t>(received) == kTotalPacketSize) {
                ReceivedAudioPacket packet;
                // ヘッダーのみをパース
                if (ParseAudioHeader(buffer.data(), kAudioHeaderBytes, packet)) {
                    // 音声ペイロードを再生キューに追加
                    packet.interleavedSamples.resize(kAudioPayloadBytes / sizeof(float));
                    std::memcpy(packet.interleavedSamples.data(), buffer.data() + kAudioHeaderBytes, kAudioPayloadBytes);

                    PacketCallback cb;
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                    cb = callback_;
                }
                if (cb) {
                    cb(packet);
                }
            }
        }
        running_ = false;
    }

    uint16_t listenPort_ = 0;
    SOCKET socket_ = INVALID_SOCKET;
    std::atomic<bool> running_{false};
    PacketCallback callback_;
    std::thread receiveThread_;
    mutable std::mutex mutex_;
    WinsockScope winsock_;
    AudioSyncPlayer audioPlayer_;
};

namespace {
    AudioUdpReceiver g_audioReceiver(kAudioListenPort);
}

bool StartAudioReceiver() {
    return g_audioReceiver.StartWithSynchronizedPlayback();
}

void StopAudioReceiver() {
    g_audioReceiver.Stop();
}

