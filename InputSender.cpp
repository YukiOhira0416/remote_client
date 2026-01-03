#include "InputSender.h"
#include "Globals.h"
#include "ReedSolomon.h"
#include "DebugLog.h"

#include <winsock2.h>
#include <ws2tcpip.h>
#include <thread>
#include <chrono>
#include <vector>

// Link with Ws2_32.lib
#pragma comment(lib, "Ws2_32.lib")

// Helper function to serialize MouseInputMessage to a byte vector
std::vector<uint8_t> SerializeMouseInput(const MouseInputMessage& msg) {
    std::vector<uint8_t> data(sizeof(MouseInputMessage));
    memcpy(data.data(), &msg, sizeof(MouseInputMessage));
    return data;
}

void InputSendThread(std::atomic<bool>& running) {
    DebugLog(L"InputSendThread started.");

    SOCKET udpSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (udpSocket == INVALID_SOCKET) {
        DebugLog(L"InputSendThread: Failed to create socket. Error: " + std::to_wstring(WSAGetLastError()));
        return;
    }

    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(INPUT_SEND_PORT);
    inet_pton(AF_INET, INPUT_SEND_IP, &serverAddr.sin_addr);

    MouseInputMessage lastMoveMsg{};
    bool hasPendingMove = false;
    auto lastSendTime = std::chrono::steady_clock::now();
    const auto sendInterval = std::chrono::milliseconds(1000 / 120); // Max 120Hz

    std::atomic<uint32_t> frameNumberCounter(0);

    while (running) {
        MouseInputMessage msg;
        bool shouldSend = false;

        // Coalesce mouse moves
        while (g_mouseInputQueue.try_dequeue(msg)) {
            if (msg.messageType == MOUSE_MOVE) {
                lastMoveMsg = msg;
                hasPendingMove = true;
            } else {
                // For non-move events, send immediately
                shouldSend = true;
                break;
            }
        }

        auto now = std::chrono::steady_clock::now();
        if (hasPendingMove && (now - lastSendTime) >= sendInterval) {
            msg = lastMoveMsg;
            hasPendingMove = false;
            shouldSend = true;
        }

        if (shouldSend) {
            std::vector<uint8_t> data = SerializeMouseInput(msg);

            std::vector<std::vector<uint8_t>> dataShards;
            std::vector<std::vector<uint8_t>> parityShards;
            size_t shard_len = 0;

            if (EncodeFEC_ISAL(data.data(), data.size(), dataShards, parityShards, shard_len, RS_K, RS_M)) {
                uint32_t frameNumber = frameNumberCounter.fetch_add(1);

                for (int i = 0; i < RS_K; ++i) {
                    ShardInfoHeader header;
                    header.frameNumber = htonl(frameNumber);
                    header.shardIndex = htonl(i);
                    header.totalDataShards = htonl(RS_K);
                    header.totalParityShards = htonl(RS_M);
                    header.originalDataLen = htonl(static_cast<uint32_t>(data.size()));

                    std::vector<uint8_t> packetData(sizeof(header) + shard_len);
                    memcpy(packetData.data(), &header, sizeof(header));
                    memcpy(packetData.data() + sizeof(header), dataShards[i].data(), shard_len);

                    sendto(udpSocket, (const char*)packetData.data(), packetData.size(), 0, (sockaddr*)&serverAddr, sizeof(serverAddr));
                }

                for (int i = 0; i < RS_M; ++i) {
                    ShardInfoHeader header;
                    header.frameNumber = htonl(frameNumber);
                    header.shardIndex = htonl(RS_K + i);
                    header.totalDataShards = htonl(RS_K);
                    header.totalParityShards = htonl(RS_M);
                    header.originalDataLen = htonl(static_cast<uint32_t>(data.size()));

                    std::vector<uint8_t> packetData(sizeof(header) + shard_len);
                    memcpy(packetData.data(), &header, sizeof(header));
                    memcpy(packetData.data() + sizeof(header), parityShards[i].data(), shard_len);

                    sendto(udpSocket, (const char*)packetData.data(), packetData.size(), 0, (sockaddr*)&serverAddr, sizeof(serverAddr));
                }
                lastSendTime = now;
            }
        }

        // Sleep briefly if idle
        if (!hasPendingMove && !shouldSend) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    closesocket(udpSocket);
    DebugLog(L"InputSendThread stopped.");
}
