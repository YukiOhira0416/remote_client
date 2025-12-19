// Audio UDP receiver that consumes packets consisting of a 32-byte header
// followed by 3,840 bytes of 16-bit stereo PCM audio (48 kHz).
//
// As long as packets continue to arrive, the payloads are enqueued for
// immediate playback. When packets stop, playback naturally drains and stops.
//
// The received bytes (header + payload) are logged in hex for inspection.

#include <winsock2.h>
#include <ws2tcpip.h>

#include <array>
#include <atomic>
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

#include "TimeSyncClient.h"
#include "AudioUdpReceiver.h"
#include "DebugLog.h"

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "winmm.lib")

struct ReceivedAudioPacket {
    std::array<uint8_t, 32> header{};         // 32-byte header (opaque)
    std::vector<int16_t> interleavedSamples;  // PCM samples (16-bit, stereo)
};

namespace {

constexpr size_t kMaxUdpPayload = 64 * 1024;  // generous buffer for UDP packets
constexpr uint16_t kAudioListenPort = 8200;
constexpr size_t kAudioHeaderSize = 32;
constexpr size_t kAudioPayloadSize = 3'840;  // PCM chunk size
constexpr uint32_t kAudioSampleRate = 48'000;
constexpr uint32_t kAudioChannels = 2;

bool ParseAudioPacket(const uint8_t* data, size_t size, ReceivedAudioPacket& out) {
    if (size < kAudioHeaderSize + kAudioPayloadSize) {
        return false;
    }

    std::memcpy(out.header.data(), data, kAudioHeaderSize);

    const size_t payloadBytes = size - kAudioHeaderSize;
    if (payloadBytes != kAudioPayloadSize) {
        return false;  // unexpected payload size
    }

    out.interleavedSamples.resize(kAudioPayloadSize / sizeof(int16_t));
    std::memcpy(out.interleavedSamples.data(), data + kAudioHeaderSize, kAudioPayloadSize);
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
        queue_.push(packet);
        cv_.notify_one();
    }

private:
    struct ActiveBuffer {
        WAVEHDR header{};
        std::vector<uint8_t> data;
    };

    void PlaybackLoop() {
        while (true) {
            ReceivedAudioPacket item;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&]() { return !running_ || !queue_.empty(); });
                if (!running_ && queue_.empty()) {
                    break;
                }
                item = queue_.front();
                queue_.pop();
            }

            SubmitPacket(item);
            CleanupCompletedHeaders();
        }
        CleanupCompletedHeaders();
    }

    bool EnsureWaveOut(uint32_t sampleRate, uint32_t channels) {
        if (waveOut_) {
            if (currentSampleRate_ == sampleRate && currentChannels_ == channels) {
                return true;
            }
            CleanupWaveOut(false);
        }

        WAVEFORMATEXTENSIBLE format{};
        format.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
        format.Format.nChannels = static_cast<WORD>(channels);
        format.Format.nSamplesPerSec = sampleRate;
        format.Format.wBitsPerSample = 16;
        format.Format.nBlockAlign = format.Format.nChannels * format.Format.wBitsPerSample / 8;
        format.Format.nAvgBytesPerSec = format.Format.nSamplesPerSec * format.Format.nBlockAlign;
        format.Format.cbSize = sizeof(WAVEFORMATEXTENSIBLE) - sizeof(WAVEFORMATEX);
        format.Samples.wValidBitsPerSample = 16;
        format.dwChannelMask = (channels == 1) ? SPEAKER_FRONT_CENTER : (SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT);
        format.SubFormat = KSDATAFORMAT_SUBTYPE_PCM;

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
        if (!EnsureWaveOut(kAudioSampleRate, kAudioChannels)) {
            return;
        }

        ActiveBuffer buffer{};
        buffer.data.resize(packet.interleavedSamples.size() * sizeof(int16_t));
        std::memcpy(buffer.data.data(), packet.interleavedSamples.data(), buffer.data.size());

        buffer.header.lpData = reinterpret_cast<LPSTR>(buffer.data.data());
        buffer.header.dwBufferLength = static_cast<DWORD>(buffer.data.size());

        if (waveOutPrepareHeader(waveOut_, &buffer.header, sizeof(WAVEHDR)) != MMSYSERR_NOERROR) {
            return;
        }
        if (waveOutWrite(waveOut_, &buffer.header, sizeof(WAVEHDR)) != MMSYSERR_NOERROR) {
            waveOutUnprepareHeader(waveOut_, &buffer.header, sizeof(WAVEHDR));
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);
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

    std::atomic<bool> running_{false};
    std::thread playbackThread_;
    std::queue<ReceivedAudioPacket> queue_;
    std::vector<ActiveBuffer> active_;
    std::mutex mutex_;
    std::condition_variable cv_;

    HWAVEOUT waveOut_ = nullptr;
    uint32_t currentSampleRate_ = 0;
    uint32_t currentChannels_ = 0;
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

            std::wstringstream logStream;
            logStream << L"[AudioUdpReceiver] Received " << received << L" bytes: ";
            logStream << std::hex << std::setw(2) << std::setfill(L'0');
            for (int i = 0; i < received; ++i) {
                logStream << std::setw(2) << static_cast<int>(buffer[i]);
                if (i + 1 < received) {
                    logStream << L" ";
                }
            }
            DebugLog(logStream.str());

            ReceivedAudioPacket packet;
            if (ParseAudioPacket(buffer.data(), static_cast<size_t>(received), packet)) {
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

