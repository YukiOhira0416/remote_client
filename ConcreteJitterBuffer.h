#pragma once
#include "JitterBuffer.h"
#include <map>
#include <mutex>
#include <condition_variable>

class ConcreteJitterBuffer : public JitterBuffer {
public:
    // The number of blocks to buffer before starting playback.
    static constexpr int JITTER_BUFFER_DEPTH = 2;

    void AddBlock(DecodedAudioBlock&& block) override {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_buffer[block.blockId] = std::move(block.pcmData);
        lock.unlock();
        m_cv.notify_one();
    }

    // Gets the next block for playback. Waits briefly if the buffer is empty.
    // Returns true if a block was retrieved, false if a timeout occurred (indicating a lost block).
    bool GetNextBlock(std::vector<uint8_t>& out_pcm, uint32_t expectedBlockId) override {
        std::unique_lock<std::mutex> lock(m_mutex);

        // Wait until the buffer has enough data to start playback or the specific block arrives.
        m_cv.wait_for(lock, std::chrono::milliseconds(15), [&] {
            return m_buffer.size() >= JITTER_BUFFER_DEPTH || m_buffer.count(expectedBlockId);
        });

        auto it = m_buffer.find(expectedBlockId);
        if (it != m_buffer.end()) {
            out_pcm = std::move(it->second);
            m_buffer.erase(it);
            return true;
        }

        // Block not found after waiting.
        return false;
    }

    bool PeekNextBlockId(uint32_t& out_blockId) override {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (m_buffer.empty()) {
            return false;
        }
        out_blockId = m_buffer.begin()->first;
        return true;
    }

private:
    std::map<uint32_t, std::vector<uint8_t>> m_buffer;
    std::mutex m_mutex;
    std::condition_variable m_cv;
};
