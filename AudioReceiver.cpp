#include "AudioReceiver.h"
#include "Crc32.h"
#include "DebugLog.h"
#include <stdexcept>

#pragma comment(lib, "ws2_32.lib")

namespace {
    // Helper to initialize Winsock once.
    void EnsureWinsockOnce() {
        static std::once_flag once;
        std::call_once(once, [] {
            WSADATA wsaData;
            if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
                throw std::runtime_error("WSAStartup failed");
            }
        });
    }
}

AudioReceiver::AudioReceiver() {
    try {
        EnsureWinsockOnce();
    } catch (const std::runtime_error& e) {
        DebugLog(L"AudioReceiver: Failed to initialize Winsock: %S", e.what());
    }
}

AudioReceiver::~AudioReceiver() {
    Stop();
    WSACleanup();
}

bool AudioReceiver::Start(int port) {
    if (m_isRunning) {
        return true; // Already running
    }

    m_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (m_socket == INVALID_SOCKET) {
        DebugLog(L"AudioReceiver: Failed to create socket, error %d", WSAGetLastError());
        return false;
    }

    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(static_cast<USHORT>(port));

    if (bind(m_socket, (SOCKADDR*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
        DebugLog(L"AudioReceiver: Failed to bind socket, error %d", WSAGetLastError());
        closesocket(m_socket);
        m_socket = INVALID_SOCKET;
        return false;
    }

    // Set a receive timeout so the thread can exit gracefully when Stop() is called.
    DWORD timeout = 500; // ms
    setsockopt(m_socket, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));

    m_isRunning = true;
    m_thread = std::thread(&AudioReceiver::ReceiverThread, this);

    DebugLog(L"AudioReceiver: Started listening on port %d", port);
    return true;
}

void AudioReceiver::Stop() {
    if (!m_isRunning) {
        return;
    }

    m_isRunning = false;
    if (m_thread.joinable()) {
        m_thread.join();
    }

    if (m_socket != INVALID_SOCKET) {
        closesocket(m_socket);
        m_socket = INVALID_SOCKET;
    }
    DebugLog(L"AudioReceiver: Stopped.");
}

bool AudioReceiver::TryDequeue(AudioShard& out_shard) {
    return m_queue.try_dequeue(out_shard);
}

void AudioReceiver::ReceiverThread() {
    std::vector<uint8_t> buffer(2048); // Sufficient for MTU

    while (m_isRunning) {
        sockaddr_in clientAddr{};
        int clientAddrSize = sizeof(clientAddr);

        int bytesReceived = recvfrom(m_socket, (char*)buffer.data(), buffer.size(), 0, (SOCKADDR*)&clientAddr, &clientAddrSize);

        if (bytesReceived == SOCKET_ERROR) {
            int error = WSAGetLastError();
            if (error == WSAETIMEDOUT) {
                continue; // Expected timeout, loop again
            }
            if (m_isRunning) {
                 DebugLog(L"AudioReceiver: recvfrom failed with error %d", error);
            }
            break; // Exit on other errors
        }

        if (bytesReceived < sizeof(AudioPacketHeader)) {
            DebugLog(L"AudioReceiver: Received a packet too small for a header.");
            continue;
        }

        AudioPacketHeader header;
        memcpy(&header, buffer.data(), sizeof(AudioPacketHeader));

        // --- Packet Validation ---
        if (memcmp(header.Magic, "AUDP", 4) != 0 || header.Version != 1 || header.HeaderSize != 64) {
            DebugLog(L"AudioReceiver: Invalid header magic, version, or size.");
            continue;
        }

        if (header.ShardTotal != header.K + header.M) {
            DebugLog(L"AudioReceiver: Inconsistent shard total. ShardTotal=%u, K=%u, M=%u", header.ShardTotal, header.K, header.M);
            continue;
        }

        const size_t payloadSize = bytesReceived - sizeof(AudioPacketHeader);
        if (payloadSize != header.ShardBytes) {
            DebugLog(L"AudioReceiver: Payload size mismatch. Expected %u, got %zu.", header.ShardBytes, payloadSize);
            continue;
        }

        uint32_t receivedHeaderCrc = header.HeaderCrc32;
        header.HeaderCrc32 = 0;
        uint32_t calculatedHeaderCrc = Crc32(&header, sizeof(AudioPacketHeader));
        if (receivedHeaderCrc != calculatedHeaderCrc) {
             DebugLog(L"AudioReceiver: Header CRC32 mismatch.");
             continue;
        }
        header.HeaderCrc32 = receivedHeaderCrc; // Restore for queueing

        const uint8_t* payloadData = buffer.data() + sizeof(AudioPacketHeader);
        uint32_t calculatedPayloadCrc = Crc32(payloadData, payloadSize);
        if (header.PayloadCrc32 != calculatedPayloadCrc) {
             DebugLog(L"AudioReceiver: Payload CRC32 mismatch.");
             continue;
        }

        // --- Validation Passed ---
        AudioShard shard;
        shard.header = header;
        shard.payload.assign(payloadData, payloadData + payloadSize);

        m_queue.enqueue(std::move(shard));
    }
}
