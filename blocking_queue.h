#ifndef BLOCKING_QUEUE_H
#define BLOCKING_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>

template<
    typename T,
    typename Container = std::vector<T>,
    typename Compare = std::less<typename Container::value_type>
>
class BlockingPriorityQueue {
public:
    void enqueue(T item) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_pq.push(std::move(item));
        }
        m_cv.notify_one();
    }

    // This new method will block until an item is available.
    // Returns true if an item was dequeued, false if shutting down.
    bool wait_and_dequeue(T& item_out) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait(lock, [this] { return !m_pq.empty() || !m_running; });
        if (!m_running && m_pq.empty()) {
            return false;
        }
        item_out = std::move(m_pq.top());
        m_pq.pop();
        return true;
    }

    size_t size_approx() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_pq.size();
    }

    bool try_dequeue(T& item_out) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_pq.empty() || !m_running) {
            return false;
        }
        item_out = std::move(m_pq.top());
        m_pq.pop();
        return true;
    }

    // Add a method to stop the wait
    void stop() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_running = false;
        }
        m_cv.notify_all();
    }

private:
    mutable std::mutex m_mutex;
    std::condition_variable m_cv;
    std::priority_queue<T, Container, Compare> m_pq;
    bool m_running = true;
};

#endif // BLOCKING_QUEUE_H
