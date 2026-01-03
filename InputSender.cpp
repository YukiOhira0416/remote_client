#include "InputSender.h"
#include "ReedSolomon.h"
#include "DebugLog.h"
#include <ws2tcpip.h>
#include <chrono>
#include <algorithm> // for std::max

InputSender::InputSender() : m_running(false), m_socket(INVALID_SOCKET) {
}

InputSender::~InputSender() {
    Stop();
}

bool InputSender::Start() {
    m_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if (m_socket == INVALID_SOCKET) {
        DebugLog(L"InputSender: socket creation failed with error: " + std::to_wstring(WSAGetLastError()));
        return false;
    }

    memset(&m_serverAddr, 0, sizeof(m_serverAddr));
    m_serverAddr.sin_family = AF_INET;
    m_serverAddr.sin_port = htons(INPUT_SEND_PORT);
    if (inet_pton(AF_INET, INPUT_SEND_IP, &m_serverAddr.sin_addr) <= 0) {
        DebugLog(L"InputSender: inet_pton failed.");
        closesocket(m_socket);
        m_socket = INVALID_SOCKET;
        return false;
    }

    m_running = true;
    m_thread = std::thread(&InputSender::SendThread, this);

    DebugLog(L"InputSender started.");
    return true;
}

void InputSender::Stop() {
    m_running = false;
    if (m_thread.joinable()) {
        m_thread.join();
    }
    if (m_socket != INVALID_SOCKET) {
        closesocket(m_socket);
        m_socket = INVALID_SOCKET;
    }
    DebugLog(L"InputSender stopped.");
}

void InputSender::EnqueueMessage(const MouseInputMessage& msg) {
    m_queue.enqueue(msg);
}

void InputSender::SendThread() {
    static std::atomic<uint32_t> frameCounter = 0;
    const auto sendInterval = std::chrono::milliseconds(1000 / 120); // 120Hz
    auto lastSendTime = std::chrono::steady_clock::now();

    // RS パラメータは固定
    const int rs_k = 14;
    const int rs_m = 8;

    while (m_running) {
        MouseInputMessage msg;
        bool hasEvents = false;
        bool hasAggregatedMove = false;

        MouseInputMessage aggregatedMsg = {0};
        aggregatedMsg.magic = MOUSE_MAGIC;
        aggregatedMsg.version = 1;

        // Process all available events in the queue
        while (m_queue.try_dequeue(msg)) {
            hasEvents = true;

            // Always update to the latest button state from any message
            aggregatedMsg.buttonsState = msg.buttonsState;

            // Coalesce MOVE events by overwriting position
            if (msg.flags & MOVE) {
                aggregatedMsg.x = msg.x;
                aggregatedMsg.y = msg.y;
                aggregatedMsg.flags |= HAS_POS | MOVE;
                hasAggregatedMove = true;
            } else {
                 // For non-MOVE events, accumulate flags and data
                aggregatedMsg.flags |= msg.flags;
                // If the event has a position, update our aggregated position
                if (msg.flags & HAS_POS) {
                    aggregatedMsg.x = msg.x;
                    aggregatedMsg.y = msg.y;
                }
                // Accumulate wheel movements
                if (msg.flags & WHEEL_V) aggregatedMsg.wheelV += msg.wheelV;
                if (msg.flags & WHEEL_H) aggregatedMsg.wheelH += msg.wheelH;
            }
        }

        auto now = std::chrono::steady_clock::now();
        bool isTimeToSend = (now - lastSendTime >= sendInterval);
        // Send if we have non-MOVE events, or if it's time to send an aggregated MOVE
        bool shouldSendNow = (aggregatedMsg.flags != 0 && aggregatedMsg.flags != (HAS_POS | MOVE)) ||
                               (hasAggregatedMove && isTimeToSend);

        if (shouldSendNow) {
            aggregatedMsg.seq = frameCounter.fetch_add(1);

            std::vector<uint8_t> data(sizeof(MouseInputMessage));
            memcpy(data.data(), &aggregatedMsg, sizeof(MouseInputMessage));

            std::vector<std::vector<uint8_t>> dataShards;
            std::vector<std::vector<uint8_t>> parityShards;
            size_t shard_len;

            // Ensure ReedSolomon.h defines this function correctly.
            // Assuming it is declared as:
            // bool EncodeFEC_ISAL(const uint8_t* data, size_t data_len, std::vector<std::vector<uint8_t>>& dataShards, std::vector<std::vector<uint8_t>>& parityShards, size_t& shard_len, int k, int m);
            if (EncodeFEC_ISAL(data.data(), data.size(), dataShards, parityShards, shard_len, rs_k, rs_m)) {
                ShardInfoHeader header;
                header.frameNumber = htonl(aggregatedMsg.seq);
                header.totalDataShards = htonl(rs_k);
                header.totalParityShards = htonl(rs_m);
                header.originalDataLen = htonl(static_cast<uint32_t>(data.size()));

                auto send_shard = [&](uint32_t index, const std::vector<uint8_t>& shard) {
                    header.shardIndex = htonl(index);
                    std::vector<uint8_t> packet(sizeof(header));
                    memcpy(packet.data(), &header, sizeof(header));
                    packet.insert(packet.end(), shard.begin(), shard.end());
                    sendto(m_socket, (const char*)packet.data(), (int)packet.size(), 0, (const sockaddr*)&m_serverAddr, sizeof(m_serverAddr));
                };

                for (int i = 0; i < rs_k; ++i) send_shard(i, dataShards[i]);
                for (int i = 0; i < rs_m; ++i) send_shard(rs_k + i, parityShards[i]);

            } else {
                DebugLog(L"InputSender: EncodeFEC_ISAL failed.");
            }

            hasAggregatedMove = false;
            lastSendTime = now;
        }

        // If no events were processed, sleep briefly to avoid busy-waiting
        if (!hasEvents) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}
