#include "DebugLog.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <processthreadsapi.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <cstdio>

#include "concurrentqueue/concurrentqueue.h" // 既にプロジェクトで利用

namespace {

using Clock = std::chrono::system_clock;

struct LogMsg {
    std::wstring text;
};

std::atomic<bool>     g_initialized{false};
std::atomic<bool>     g_running{false};
std::atomic<bool>     g_enabled{true};
std::atomic<int>      g_seq{0};
std::atomic<size_t>   g_qSize{0};
std::atomic<size_t>   g_dropped{0};

size_t                g_capacity = 8192;
bool                  g_alsoODS  = true;

moodycamel::ConcurrentQueue<LogMsg> g_queue;

std::wofstream        g_file;
std::thread*          g_worker = nullptr;
std::mutex            g_wakeupMtx;
std::condition_variable g_wakeupCv;

// 実行ファイルディレクトリ + ファイル名
std::wstring MakeLogPath(const wchar_t* fileName) noexcept {
    wchar_t exePath[MAX_PATH]{};
    DWORD n = GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) {
        // フォールバック：カレントディレクトリ
        return std::wstring(fileName ? fileName : L"debuglog_client.log");
    }
    std::wstring path(exePath);
    // ディレクトリ部分へ
    auto pos = path.find_last_of(L"\\/");
    if (pos != std::wstring::npos) path.erase(pos + 1);
    path += (fileName ? fileName : L"debuglog_client.log");
    return path;
}

inline DWORD GetThreadIdFast() noexcept {
    return ::GetCurrentThreadId();
}

inline std::wstring NowString() {
    const auto now = Clock::now();
    const auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    std::time_t t  = Clock::to_time_t(now);
    std::tm lt{};
    localtime_s(&lt, &t);
    wchar_t buf[64]{};
    std::swprintf(buf, 64, L"%02d:%02d:%02d.%03d", lt.tm_hour, lt.tm_min, lt.tm_sec, static_cast<int>(ms.count()));
    return buf;
}

void WriteLine(const std::wstring& w) {
    if (g_file.is_open()) {
        g_file << w << L"\n";
    }
}

void WorkerLoop() {
    // 書き込みのバッチング設定
    constexpr size_t FLUSH_EVERY_N = 64;
    constexpr auto   FLUSH_EVERY_T = std::chrono::milliseconds(200);

    auto lastFlush = Clock::now();
    size_t buffered = 0;

    while (g_running.load(std::memory_order_acquire) || g_qSize.load(std::memory_order_relaxed) > 0) {
        LogMsg msg;
        bool got = g_queue.try_dequeue(msg);
        if (!got) {
            // 何も無ければ少し待つ
            std::unique_lock<std::mutex> lk(g_wakeupMtx);
            g_wakeupCv.wait_for(lk, std::chrono::milliseconds(25));
            continue;
        }
        g_qSize.fetch_sub(1, std::memory_order_relaxed);

        // 先に、前回までにドロップがあればまとめて1行出す
        size_t d = g_dropped.exchange(0, std::memory_order_acq_rel);
        if (d > 0) {
            std::wstring note = L"[LOG] dropped " + std::to_wstring(d) + L" messages due to queue overflow";
            WriteLine(note);
            ++buffered;
        }

        WriteLine(msg.text);
        ++buffered;

        const auto now = Clock::now();
        if (buffered >= FLUSH_EVERY_N || (now - lastFlush) >= FLUSH_EVERY_T) {
            if (g_file.is_open()) g_file.flush();
            buffered = 0;
            lastFlush = now;
        }
    }

    // 最後に完全 flush
    if (g_file.is_open()) g_file.flush();
}

} // anonymous namespace

bool DebugLogAsync::Init(const wchar_t* logFileName, size_t queueCapacity, bool alsoOutputDebugString) noexcept {
    bool expected = false;
    if (!g_initialized.compare_exchange_strong(expected, true)) {
        return true; // 既に初期化済み
    }

    g_capacity = (queueCapacity == 0 ? 8192 : queueCapacity);
    g_alsoODS = alsoOutputDebugString;

    const std::wstring logPath = MakeLogPath(logFileName);

    // UTF-16LE (BOMなし) のまま書く。既存も wofstream で書いていたので互換。
    g_file.open(logPath, std::ios::out | std::ios::app);
    // 失敗してもODSでの運用は継続する

    g_running.store(true, std::memory_order_release);
    try {
        g_worker = new std::thread(&WorkerLoop);
    } catch (...) {
        g_running.store(false, std::memory_order_release);
        if (g_file.is_open()) g_file.close();
        return false;
    }

    return g_file.is_open();
}

void DebugLogAsync::Shutdown() noexcept {
    if (!g_initialized.load(std::memory_order_acquire)) return;
    bool wasRunning = g_running.exchange(false, std::memory_order_acq_rel);
    // 起床させる
    g_wakeupCv.notify_all();

    if (wasRunning && g_worker && g_worker->joinable()) {
        try {
            g_worker->join();
        } catch (...) {
            // join が例外を投げても、少なくともメモリリークは防ぐ
        }
        delete g_worker;
        g_worker = nullptr;
    }

    if (g_file.is_open()) {
        try { g_file.flush(); } catch (...) {}
        try { g_file.close(); } catch (...) {}
    }
}

void DebugLogAsync::ForceFlush() noexcept {
    if (g_file.is_open()) {
        try { g_file.flush(); } catch (...) {}
    }
}

void DebugLogAsync::SetEnabled(bool enabled) noexcept {
    g_enabled.store(enabled, std::memory_order_release);
}

void DebugLog(const std::wstring& message) {
    if (!g_enabled.load(std::memory_order_acquire)) return;

    const int seq = 1 + g_seq.fetch_add(1, std::memory_order_relaxed);

    // 形式: "1234: [HH:MM:SS.mmm][tid=xxxx] message"
    std::wstring line;
    line.reserve(message.size() + 64);
    line.append(std::to_wstring(seq));
    line.append(L": [").append(NowString()).append(L"][tid=");
    line.append(std::to_wstring(static_cast<unsigned long long>(GetThreadIdFast()))).append(L"] ");
    line.append(message);

    // ODS は即時に（開発時の可視性）
    // ワーカ未初期化でも ODS は出す
    if (g_alsoODS) {
        ::OutputDebugStringW(line.c_str());
        ::OutputDebugStringW(L"\n");
    }

    if (!g_initialized.load(std::memory_order_acquire)) {
        // 未初期化ならファイルへの永続化は諦めて即return（ODSのみ）
        return;
    }

    // 容量超過ならドロップを記録
    size_t cur = g_qSize.load(std::memory_order_relaxed);
    while (true) {
        if (cur >= g_capacity) {
            g_dropped.fetch_add(1, std::memory_order_acq_rel);
            return;
        }
        if (g_qSize.compare_exchange_weak(cur, cur + 1, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            break;
        }
        // CAS失敗時はcurが更新されるのでループ続行
    }

    g_queue.enqueue(LogMsg{ std::move(line) });
    // ワーカを起床
    g_wakeupCv.notify_one();
}
