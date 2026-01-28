#pragma once

#include <QObject>
#include <QTimer>
#include <windows.h>
#include <string>
#include <vector>

// フォーカス中だけローカルの Xbox Game Bar を「実質無効」にする。
// 要件B（ローカルでトースト/HDR切替を完全に起こさない）向け。
//
// 手段:
//  1) HKCUのGameDVR/GameBar/BackgroundAccessApplicationsを書換（必ず復帰）
//  2) XboxGamingOverlay関連プロセスを定期的にKill（起動しても即落とす）
//
// 環境差対応:
//  - PFN（PackageFamilyName）は既知候補＋レジストリ走査で解決
//  - Killは PFN一致（GetPackageFamilyName）を優先し、exe名候補も併用
class GameBarSuppressor : public QObject {
    Q_OBJECT
public:
    explicit GameBarSuppressor(QObject* parent = nullptr);
    ~GameBarSuppressor();

    void setSuppressed(bool enabled); // focus ON/OFF

private:
    struct DwordBackup { bool existed=false; DWORD value=0; };

    bool m_enabled = false;
    QTimer m_killTimer;

    // バックアップ
    DwordBackup m_bk_AppCaptureEnabled;
    DwordBackup m_bk_GameDVR_Enabled;
    DwordBackup m_bk_UseNexusForGameBarEnabled;

    // BackgroundAccessApplications (PFNごとに持つ必要がある)
    struct BgBackup {
        std::wstring pfn;
        DwordBackup disabled;
        DwordBackup disabledByUser;
    };
    std::vector<BgBackup> m_bgBackups;

    // PFN候補（環境差対応）
    std::vector<std::wstring> m_packageFamilies;

    void apply();
    void restore();

    // PFN解決
    static std::vector<std::wstring> resolvePackageFamilies();
    static void dedup(std::vector<std::wstring>& v);

    // レジストリユーティリティ
    static bool readDword(HKEY root, const wchar_t* subKey, const wchar_t* valueName, DwordBackup& out);
    static bool writeDword(HKEY root, const wchar_t* subKey, const wchar_t* valueName, DWORD v);
    static bool deleteValue(HKEY root, const wchar_t* subKey, const wchar_t* valueName);

    // Kill
    void killOnce();
    static void killByPackageFamily(const std::vector<std::wstring>& pfns);
    static void killByExeNames();
};
