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

// Server expects a versioned 28-byte payload (magic/version/flags/...).
// The client MouseInputMessage in Globals.h is an INTERNAL event struct and must not be sent as-is.
namespace {
#pragma pack(push, 1)
struct MouseInputMessageWireV1 {
    uint32_t magic;
    uint16_t version;
    uint16_t flags;
    int32_t  x;
    int32_t  y;
    int16_t  wheelV;
    int16_t  wheelH;
    uint32_t buttonsState;
    uint32_t seq;
};
#pragma pack(pop)
static_assert(sizeof(MouseInputMessageWireV1) == 28, "MouseInputMessageWireV1 must be 28 bytes");

constexpr uint32_t MOUSE_INPUT_MAGIC = 'M' | ('I' << 8) | ('N' << 16) | ('1' << 24);
constexpr uint16_t MOUSE_INPUT_VERSION = 1;

// Must match server-side MouseEventFlags (server/Globals.h)
enum MouseEventFlagsWire : uint16_t {
    HAS_POS_W           = 0x0001,
    MOVE_W              = 0x0002,
    L_DOWN_W            = 0x0004,
    L_UP_W              = 0x0008,
    R_DOWN_W            = 0x0010,
    R_UP_W              = 0x0020,
    M_DOWN_W            = 0x0040,
    M_UP_W              = 0x0080,
    WHEEL_V_W           = 0x0100,
    WHEEL_H_W           = 0x0200,
    POS_IS_LAST_VALID_W = 0x0400,
};

static MouseInputMessageWireV1 BuildWireMessage(const MouseInputMessage& msg, uint32_t seq)
{
    MouseInputMessageWireV1 w{};
    w.magic = MOUSE_INPUT_MAGIC;
    w.version = MOUSE_INPUT_VERSION;

    uint16_t flags = HAS_POS_W;
    switch (msg.messageType) {
    case MOUSE_MOVE:    flags |= MOVE_W;    break;
    case LBUTTON_DOWN:  flags |= L_DOWN_W;  break;
    case LBUTTON_UP:    flags |= L_UP_W;    break;
    case RBUTTON_DOWN:  flags |= R_DOWN_W;  break;
    case RBUTTON_UP:    flags |= R_UP_W;    break;
    case MBUTTON_DOWN:  flags |= M_DOWN_W;  break;
    case MBUTTON_UP:    flags |= M_UP_W;    break;
    case WHEEL:         flags |= WHEEL_V_W; break;
    case HWHEEL:        flags |= WHEEL_H_W; break;
    default: break;
    }
    if (msg.flags & POS_IS_LAST_VALID) {
        flags |= POS_IS_LAST_VALID_W;
    }

    w.flags = flags;
    w.x = (int32_t)msg.x;
    w.y = (int32_t)msg.y;
    w.wheelV = (int16_t)msg.wheelDelta;
    w.wheelH = (int16_t)msg.wheelHDelta;
    w.buttonsState = (uint32_t)msg.buttonsState;
    w.seq = seq;
    return w;
}

static std::vector<uint8_t> SerializeWireMessage(const MouseInputMessageWireV1& w)
{
    std::vector<uint8_t> data(sizeof(MouseInputMessageWireV1));
    memcpy(data.data(), &w, sizeof(MouseInputMessageWireV1));
    return data;
}
} // namespace

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

    // 1 input message == 1 FEC frame. Use the same monotonic counter for both seq and frameNumber.
    std::atomic<uint32_t> messageCounter(0);

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
            const uint32_t seq = messageCounter.fetch_add(1, std::memory_order_relaxed);
            const MouseInputMessageWireV1 wire = BuildWireMessage(msg, seq);
            std::vector<uint8_t> data = SerializeWireMessage(wire);

            std::vector<std::vector<uint8_t>> dataShards;
            std::vector<std::vector<uint8_t>> parityShards;
            size_t shard_len = 0;

            if (EncodeFEC_ISAL(data.data(), data.size(), dataShards, parityShards, shard_len, RS_K, RS_M)) {
                const uint32_t frameNumber = seq;

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
