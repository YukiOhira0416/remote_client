#ifndef INPUT_SENDER_H
#define INPUT_SENDER_H

#include <thread>
#include <atomic>
#include <vector>
#include <winsock2.h>
#include "Globals.h"
#include "concurrentqueue.h"

// Flags for MouseInputMessage (must match server)
#define MOUSE_MAGIC 0x4d494e31
#define HAS_POS (1 << 0)
#define MOVE (1 << 1)
#define L_DOWN (1 << 2)
#define L_UP (1 << 3)
#define R_DOWN (1 << 4)
#define R_UP (1 << 5)
#define WHEEL_V (1 << 6)
#define WHEEL_H (1 << 7)
#define POS_IS_LAST_VALID (1 << 8)

// This must match the server-side definition
struct MouseInputMessage {
    uint32_t magic;
    uint16_t version;
    uint16_t flags;
    int32_t x;
    int32_t y;
    int16_t wheelV;
    int16_t wheelH;
    uint32_t buttonsState;
    uint32_t seq;
};


class InputSender {
public:
    InputSender();
    ~InputSender();

    bool Start();
    void Stop();
    void EnqueueMessage(const MouseInputMessage& msg);

private:
    void SendThread();

    std::atomic<bool> m_running;
    std::thread m_thread;
    moodycamel::ConcurrentQueue<MouseInputMessage> m_queue;

    SOCKET m_socket;
    sockaddr_in m_serverAddr;
};

#endif // INPUT_SENDER_H
