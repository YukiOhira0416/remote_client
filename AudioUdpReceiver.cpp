// Standalone audio UDP receiver for packets produced by AudioUdpSender.
//
// This file is intentionally self contained so it can be copied to a client
// project as-is. It is not part of the server build; compile manually when
// needed (e.g., `cl /std:c++17 /EHsc AudioUdpReceiverSample.cpp ws2_32.lib`).
//
// Packet layout:
// [32-byte header (unused)][3840 bytes of interleaved PCM16 stereo samples]
// - The payload represents 960 stereo frames (20 ms at 48 kHz, 16-bit).

#include <winsock2.h>
#include <ws2tcpip.h>

#include <atomic>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <queue>
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
    // Interleaved signed 16-bit PCM samples (stereo).
    std::vector<int16_t> interleavedSamples;
};

namespace {

constexpr size_t kMaxUdpPayload = 64 * 1024;  // generous buffer for UDP packets
constexpr uint16_t kAudioListenPort = 8200;
constexpr size_t kAudioHeaderSize = 32;       // 32-byte header before audio payload
constexpr size_t kAudioPayloadSize = 3840;    // 960 stereo frames @ 16-bit PCM (20 ms @ 48kHz)
constexpr uint32_t kAudioSampleRate = 48'000;
constexpr uint32_t kAudioChannels = 2;
constexpr size_t kExpectedPacketSize = kAudioHeaderSize + kAudioPayloadSize;

struct QueuedAudioPacket {
    ReceivedAudioPacket packet;
};

bool ParseAudioPacket(const uint8_t* data, size_t size, ReceivedAudioPacket& out) {
    if (size != kExpectedPacketSize) {
        return false;
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
        queue_.push({packet});
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
                cv_.wait(lock, [&]() { return !running_ || !queue_.empty(); });
                if (!running_ && queue_.empty()) {
                    break;
                }
                item = queue_.front();
                queue_.pop();
            }

            SubmitPacket(item.packet);
            CleanupCompletedHeaders();
        }
        CleanupCompletedHeaders();
    }

    bool EnsureWaveOut() {
        if (waveOut_) {
            return true;
        }

        WAVEFORMATEX format{};
        format.wFormatTag = WAVE_FORMAT_PCM;
        format.nChannels = static_cast<WORD>(kAudioChannels);
        format.nSamplesPerSec = kAudioSampleRate;
        format.wBitsPerSample = 16;
        format.nBlockAlign = format.nChannels * format.wBitsPerSample / 8;
        format.nAvgBytesPerSec = format.nSamplesPerSec * format.nBlockAlign;

        MMRESULT res = waveOutOpen(&waveOut_, WAVE_MAPPER, &format, 0, 0, CALLBACK_NULL);
        if (res != MMSYSERR_NOERROR) {
            waveOut_ = nullptr;
            return false;
        }

        currentSampleRate_ = kAudioSampleRate;
        currentChannels_ = kAudioChannels;
        return true;
    }

    void SubmitPacket(const ReceivedAudioPacket& packet) {
        if (!EnsureWaveOut()) {
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
    std::queue<QueuedAudioPacket> queue_;
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

        // Stop playback if no packets arrive for a short period.
        const DWORD recvTimeoutMs = 2000;
        setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&recvTimeoutMs), sizeof(recvTimeoutMs));

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
        std::thread joinTarget;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            running_ = false;
            if (socket_ != INVALID_SOCKET) {
                closesocket(socket_);
                socket_ = INVALID_SOCKET;
            }
            joinTarget = std::move(receiveThread_);
        }
        if (joinTarget.joinable()) {
            joinTarget.join();
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
                const int err = WSAGetLastError();
                if (err == WSAETIMEDOUT) {
                    DebugLog(L"[AudioUdpReceiver] No audio data received for timeout window; stopping playback.");
                }
                break;  // socket closed or error; exit loop and let Stop() clean up
            }

            if (received != static_cast<int>(kExpectedPacketSize)) {
                DebugLog(L"[AudioUdpReceiver] Dropping packet with unexpected size.");
                continue;
            }

            const uint8_t* payloadBegin = buffer.data() + kAudioHeaderSize;
            const uint8_t* payloadEnd = payloadBegin + kAudioPayloadSize;
            if (std::all_of(payloadBegin, payloadEnd, [](uint8_t b) { return b == 0; })) {
                DebugLog(L"[AudioUdpReceiver] Dropping packet with zeroed audio payload.");
                continue;
            }

            DebugLog(L"[AudioUdpReceiver] Received audio packet.");

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
        audioPlayer_.Stop();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (socket_ != INVALID_SOCKET) {
                closesocket(socket_);
                socket_ = INVALID_SOCKET;
            }
            callback_ = nullptr;
        }
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

