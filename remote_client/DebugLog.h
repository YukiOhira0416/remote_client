#pragma once
#include <string>

/// 既存互換API（非同期キューに投入するだけ）
void DebugLog(const std::wstring& message);

namespace DebugLogAsync {

    /// ログ・ワーカースレッド初期化
    /// @param logFileName  実行ファイルディレクトリ直下に作成されるログファイル名
    /// @param queueCapacity ドロップを開始するおおよその上限（メッセージ数）
    /// @param alsoOutputDebugString OutputDebugStringWへの即時出力を行うか
    /// @return ファイルオープンに成功した場合 true（失敗してもODSには出る）
    bool Init(const wchar_t* logFileName = L"debuglog_client.log",
              size_t queueCapacity = 8192,
              bool alsoOutputDebugString = true) noexcept;

    /// バッファを全て書き出して終了。二重呼び出しは安全（idempotent）。
    void Shutdown() noexcept;

    /// 明示的にflush（テストや緊急時用）
    void ForceFlush() noexcept;

    /// ランタイムでの有効/無効切り替え（falseならキュー投入もしない）
    void SetEnabled(bool enabled) noexcept;

} // namespace DebugLogAsync
