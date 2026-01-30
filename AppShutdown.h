#pragma once
#include <thread>
#include <vector>
#include <atomic>

struct AppThreads {
    std::thread* bandwidthThread = nullptr;
    std::thread* resendThread = nullptr;
    std::vector<std::thread>* receiverThreads = nullptr;
    std::vector<std::thread>* fecWorkerThreads = nullptr;
    std::vector<std::thread>* nvdecThreads = nullptr;
    std::thread* rebootListenerThread = nullptr;
    std::thread* inputSenderThread = nullptr;
    std::atomic<bool>* input_sender_running = nullptr;
    std::thread* frameMonitorThread = nullptr;
};

// Asynchronous signal: sets flags and, if necessary, exits the message loop.
// appRunningPtr is optional (can be nullptr). If provided, it writes false.
void RequestShutdown(std::atomic<bool>* appRunningPtr = nullptr);

// Synchronous shutdown: will only be executed once (idempotent).
void ReleaseAllResources(const AppThreads& threads);
