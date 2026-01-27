#include "KeyboardSender.h"
#include "Globals.h"
#include "ReedSolomon.h"
#include "concurrentqueue/concurrentqueue.h"
#include "DebugLog.h"

#include <winsock2.h>
#include <ws2tcpip.h>
#include <unordered_set>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>
#include <string>

// Link with Ws2_32.lib
#pragma comment(lib, "Ws2_32.lib")

// Internal queue
namespace {
enum class MsgType : uint8_t { Event = 1, Focus = 2 };

struct Msg {
    MsgType type;
    uint8_t active;     // for focus
    uint16_t makeCode;  // for event
    uint16_t flags;     // WIRE flags (not RAW flags)
    uint16_t vkey;      // Virtual Key (0=unknown)
};

moodycamel::ConcurrentQueue<Msg> g_q;

// WIRE flags (Matching server expectations)
constexpr uint16_t KBD_KEYUP = 0x0001;
constexpr uint16_t KBD_E0    = 0x0002;
constexpr uint16_t KBD_E1    = 0x0004;

// Convert RAWKEYBOARD.Flags to wire flags
static uint16_t RawToWireFlags(uint16_t rawFlags)
{
    uint16_t f = 0;
    if (rawFlags & RI_KEY_BREAK) f |= KBD_KEYUP;
    if (rawFlags & RI_KEY_E0)    f |= KBD_E0;
    if (rawFlags & RI_KEY_E1)    f |= KBD_E1;
    return f;
}

static inline uint64_t MakeKeyId(uint16_t makeCode, uint16_t flagsNoKeyUp, uint16_t vkey)
{
    const uint16_t f = (uint16_t)(flagsNoKeyUp & (KBD_E0 | KBD_E1));
    return ((uint64_t)vkey << 32) | ((uint64_t)f << 16) | (uint64_t)makeCode;
}

#pragma pack(push, 1)
struct KeyboardWireHeaderV1 {
    uint32_t magic;    // 'K''I''N''1'
    uint16_t version;  // 1
    uint16_t msgType;  // 1=EVENT, 2=STATE_SYNC
    uint32_t seq;
};
struct KeyboardWireEventV2 {
    uint16_t makeCode;
    uint16_t flags;    // KEYUP/E0/E1
    uint16_t vkey;     // Virtual Key
    uint16_t reserved;
};
struct KeyboardWireStateV2 {
    uint16_t count;
    uint16_t reserved;
    // followed by count * KeyboardWireEventV2 (KEYUP bit must be 0)
};
#pragma pack(pop)

constexpr uint32_t KEYBOARD_MAGIC   = 'K' | ('I' << 8) | ('N' << 16) | ('1' << 24);
constexpr uint16_t KEYBOARD_VERSION = 2;
constexpr uint16_t MSG_EVENT        = 1;
constexpr uint16_t MSG_STATE_SYNC   = 2;

static std::vector<uint8_t> BuildEventPayload(uint32_t seq, uint16_t makeCode, uint16_t flags, uint16_t vkey)
{
    KeyboardWireHeaderV1 h{};
    h.magic   = KEYBOARD_MAGIC;
    h.version = KEYBOARD_VERSION;
    h.msgType = MSG_EVENT;
    h.seq     = seq;

    KeyboardWireEventV2 e{};
    e.makeCode = makeCode;
    e.flags    = flags;
    e.vkey     = vkey;
    e.reserved = 0;

    std::vector<uint8_t> out(sizeof(h) + sizeof(e));
    memcpy(out.data(), &h, sizeof(h));
    memcpy(out.data() + sizeof(h), &e, sizeof(e));
    return out;
}

static std::vector<uint8_t> BuildStatePayload(uint32_t seq, const std::unordered_set<uint64_t>& pressed)
{
    KeyboardWireHeaderV1 h{};
    h.magic   = KEYBOARD_MAGIC;
    h.version = KEYBOARD_VERSION;
    h.msgType = MSG_STATE_SYNC;
    h.seq     = seq;

    KeyboardWireStateV2 st{};
    st.count = (uint16_t)pressed.size();
    st.reserved = 0;

    std::vector<uint8_t> out;
    out.resize(sizeof(h) + sizeof(st) + pressed.size() * sizeof(KeyboardWireEventV2));

    memcpy(out.data(), &h, sizeof(h));
    memcpy(out.data() + sizeof(h), &st, sizeof(st));

    auto* p = out.data() + sizeof(h) + sizeof(st);
    for (const auto& id : pressed) {
        KeyboardWireEventV2 e{};
        e.makeCode = (uint16_t)(id & 0xFFFF);
        e.flags    = (uint16_t)((id >> 16) & 0xFFFF); // No KEYUP bit
        e.vkey     = (uint16_t)((id >> 32) & 0xFFFF);
        e.reserved = 0;
        memcpy(p, &e, sizeof(e));
        p += sizeof(e);
    }
    return out;
}

static bool SendWithFEC(SOCKET s, const sockaddr_in& dst, uint32_t frameNumber, const std::vector<uint8_t>& payload)
{
    std::vector<std::vector<uint8_t>> dataShards, parityShards;
    size_t shard_len = 0;

    if (!EncodeFEC_ISAL(payload.data(), payload.size(), dataShards, parityShards, shard_len, RS_K, RS_M)) {
        return false;
    }

    // data shards
    for (int i = 0; i < RS_K; ++i) {
        ShardInfoHeader header{};
        header.frameNumber       = htonl(frameNumber);
        header.shardIndex        = htonl((uint32_t)i);
        header.totalDataShards   = htonl((uint32_t)RS_K);
        header.totalParityShards = htonl((uint32_t)RS_M);
        header.originalDataLen   = htonl((uint32_t)payload.size());

        std::vector<uint8_t> pkt(sizeof(header) + shard_len);
        memcpy(pkt.data(), &header, sizeof(header));
        memcpy(pkt.data() + sizeof(header), dataShards[i].data(), shard_len);

        sendto(s, (const char*)pkt.data(), (int)pkt.size(), 0, (const sockaddr*)&dst, sizeof(dst));
    }

    // parity shards
    for (int i = 0; i < RS_M; ++i) {
        ShardInfoHeader header{};
        header.frameNumber       = htonl(frameNumber);
        header.shardIndex        = htonl((uint32_t)(RS_K + i));
        header.totalDataShards   = htonl((uint32_t)RS_K);
        header.totalParityShards = htonl((uint32_t)RS_M);
        header.originalDataLen   = htonl((uint32_t)payload.size());

        std::vector<uint8_t> pkt(sizeof(header) + shard_len);
        memcpy(pkt.data(), &header, sizeof(header));
        memcpy(pkt.data() + sizeof(header), parityShards[i].data(), shard_len);

        sendto(s, (const char*)pkt.data(), (int)pkt.size(), 0, (const sockaddr*)&dst, sizeof(dst));
    }

    return true;
}
} // namespace

void EnqueueKeyboardRawEvent(uint16_t makeCode, uint16_t rawFlags, uint16_t vkey)
{
    // NOTE:
    //   以前は Windowsキー(LWIN/RWIN) をローカルOS側で処理させるため、ここで送信を抑止していた。
    //   今回は「クライアントがフォーカス中はローカルでWinキーを動作させず、リモートへ送る」方針のため
    //   Winキーも他キー同様に送信キューへ投入する。

    Msg m{};
    m.type = MsgType::Event;
    m.active = 0;
    m.makeCode = makeCode;
    m.flags = RawToWireFlags(rawFlags);
    m.vkey  = vkey;
    g_q.enqueue(m);
}

void EnqueueKeyboardFocusChanged(bool active)
{
    Msg m{};
    m.type = MsgType::Focus;
    m.active = active ? 1 : 0;
    m.makeCode = 0;
    m.flags = 0;
    g_q.enqueue(m);
}

void KeyboardSendThread(std::atomic<bool>& running)
{
    DebugLog(L"KeyboardSendThread started.");

    SOCKET sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock == INVALID_SOCKET) {
        DebugLog(L"KeyboardSendThread: socket() failed: " + std::to_wstring(WSAGetLastError()));
        return;
    }

    // bind source: 192.168.0.3:8400
    sockaddr_in local{};
    local.sin_family = AF_INET;
    local.sin_port = htons(KEYBOARD_BIND_PORT);
    if (inet_pton(AF_INET, KEYBOARD_BIND_IP, &local.sin_addr) <= 0) {
        DebugLog(L"KeyboardSendThread: invalid KEYBOARD_BIND_IP.");
        closesocket(sock);
        return;
    }
    if (bind(sock, (sockaddr*)&local, sizeof(local)) == SOCKET_ERROR) {
        DebugLog(L"KeyboardSendThread: bind() failed: " + std::to_wstring(WSAGetLastError()));
        closesocket(sock);
        return;
    }

    // destination: 192.168.0.2:8400
    sockaddr_in dst{};
    dst.sin_family = AF_INET;
    dst.sin_port = htons(KEYBOARD_SEND_PORT);
    inet_pton(AF_INET, KEYBOARD_SEND_IP, &dst.sin_addr);

    bool active = false;
    std::unordered_set<uint64_t> pressed;
    uint32_t seq = 0;

    auto lastSync = std::chrono::steady_clock::now();
    auto lastActivity = std::chrono::steady_clock::now();
    const auto syncInterval = std::chrono::milliseconds(30);

    while (running.load()) {
        Msg msg{};
        bool didWork = false;

        while (g_q.try_dequeue(msg)) {
            didWork = true;

            if (msg.type == MsgType::Focus) {
                active = (msg.active != 0);
                // We no longer clear 'pressed' on focus loss to maintain sync with physical keys.
                // We also no longer send an empty SYNC immediately.
                lastSync = std::chrono::steady_clock::now();
                lastActivity = lastSync;
                continue;
            }

            if (!active) {
                // Ignore new key-downs when not active, but allow key-ups for keys that are currently pressed.
                const bool isUp = (msg.flags & KBD_KEYUP) != 0;
                const uint64_t id = MakeKeyId(msg.makeCode, msg.flags, msg.vkey);
                if (!isUp || pressed.find(id) == pressed.end()) {
                    continue;
                }
            }

            // EVENT
            const uint32_t fnum = seq++;
            auto payload = BuildEventPayload(fnum, msg.makeCode, msg.flags, msg.vkey);
            SendWithFEC(sock, dst, fnum, payload);

            // Update pressed state
            const bool isUp = (msg.flags & KBD_KEYUP) != 0;
            const uint64_t id = MakeKeyId(msg.makeCode, msg.flags, msg.vkey);
            if (isUp) pressed.erase(id);
            else      pressed.insert(id);

            lastActivity = std::chrono::steady_clock::now();
        }

        const auto now = std::chrono::steady_clock::now();
        // Periodic STATE_SYNC to handle packet loss.
        // We continue syncing even when inactive if there are keys currently held down.
        if ((now - lastSync) >= syncInterval) {
            bool shouldSync = false;
            const bool recent = (now - lastActivity) < std::chrono::seconds(1);
            if (active) {
                if (!pressed.empty() || recent) shouldSync = true;
            } else {
                if (!pressed.empty()) shouldSync = true;
            }

            if (shouldSync) {
                const uint32_t fnum = seq++;
                auto payload = BuildStatePayload(fnum, pressed);
                SendWithFEC(sock, dst, fnum, payload);
                lastSync = now;
            }
        }

        if (!didWork) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    closesocket(sock);
    DebugLog(L"KeyboardSendThread exiting.");
}
