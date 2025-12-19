#pragma once
#include <winsock2.h>
#include <ws2tcpip.h>
#include <thread>
#include <atomic>
#include <vector>
#include "concurrentqueue/concurrentqueue.h"
#include "AudioPacketTypes.h"

// Struct to hold a validated shard and its payload, ready for the decoder.
struct AudioShard {
    AudioPacketHeader header;
    std::vector<uint8_t> payload;
};

class AudioReceiver {
public:
    AudioReceiver();
    ~AudioReceiver();

    // Starts the receiver thread to listen on the specified port.
    bool Start(int port = 8200);

    // Stops the receiver thread and cleans up resources.
    void Stop();

    // Tries to dequeue a validated audio shard. Returns true on success.
    bool TryDequeue(AudioShard& out_shard);

private:
    void ReceiverThread();

    std::atomic<bool> m_isRunning{false};
    SOCKET m_socket{INVALID_SOCKET};
    std::thread m_thread;
    moodycamel::ConcurrentQueue<AudioShard> m_queue;
};
