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
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#pragma comment(lib, "ws2_32.lib")

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

bool ParseAudioPacket(const uint8_t* data, size_t size, ReceivedAudioPacket& out) {
    const size_t headerSize = sizeof(uint64_t) * 2 + sizeof(uint32_t) * 4;
    if (size < headerSize) {
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

    uint32_t sampleCount = 0;
    read_u32(sampleCount);

    const size_t expectedSize = headerSize + static_cast<size_t>(sampleCount) * sizeof(float);
    if (sampleCount == 0 || size < expectedSize) {
        return false;
    }

    out.interleavedSamples.resize(sampleCount);
    std::memcpy(out.interleavedSamples.data(), data + offset, sampleCount * sizeof(float));
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

    void Stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_) {
            return;
        }
        running_ = false;
        if (socket_ != INVALID_SOCKET) {
            closesocket(socket_);
            socket_ = INVALID_SOCKET;
        }
        if (receiveThread_.joinable()) {
            receiveThread_.join();
        }
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
};

