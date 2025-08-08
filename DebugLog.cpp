#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING

#include "DebugLog.h"
#include <windows.h>
#include <fstream>
#include <iostream>
#include <experimental/filesystem>
#include <atomic>
#include <mutex>

// ログの呼び出し回数をカウントする静的変数
static std::atomic<int> logCounter(0);

// ファイル出力時の排他制御用ミューテックス
static std::mutex logMutex;

void DebugLog(const std::wstring& message) {
    int logNumber = ++logCounter;
    std::wstring numberedMessage = std::to_wstring(logNumber) + L": " + message;

    OutputDebugStringW(numberedMessage.c_str());

    std::lock_guard<std::mutex> lock(logMutex);

    char exePath[MAX_PATH];
    if (GetModuleFileNameA(NULL, exePath, MAX_PATH) == 0) {
        std::string errorMessage = "Failed to get module file name: " + std::to_string(GetLastError());
        OutputDebugStringA(errorMessage.c_str());
        return;
    }

    std::experimental::filesystem::path logFilePath(exePath);
    logFilePath.remove_filename();
    logFilePath /= "debuglog_client.log";

    std::wofstream logFile(logFilePath, std::ios::app);
    if (logFile.is_open()) {
        logFile << numberedMessage << std::endl << std::flush;  // 即座に flush を追加
        logFile.close();
    }
    else {
        std::string errorMessage = "Failed to open log file: " + logFilePath.string();
        OutputDebugStringA(errorMessage.c_str());
    }
}
