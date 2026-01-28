#include "GameBarSuppressor.h"
#include "DebugLog.h"

#include <tlhelp32.h>
#include <cwctype>
#include <mutex>

// GetPackageFamilyName
#include <appmodel.h>

static bool ieq(const std::wstring& a, const std::wstring& b) {
    if (a.size() != b.size()) return false;
    for (size_t i=0;i<a.size();++i) {
        if (towlower(a[i]) != towlower(b[i])) return false;
    }
    return true;
}

static bool containsI(const std::wstring& s, const std::wstring& pat) {
    if (pat.empty()) return false;
    for (size_t i=0;i+pat.size()<=s.size();++i) {
        bool ok=true;
        for (size_t j=0;j<pat.size();++j) {
            if (towlower(s[i+j]) != towlower(pat[j])) { ok=false; break; }
        }
        if (ok) return true;
    }
    return false;
}

static std::wstring bgKeyPath(const std::wstring& pfn) {
    return L"Software\\Microsoft\\Windows\\CurrentVersion\\BackgroundAccessApplications\\" + pfn;
}

GameBarSuppressor::GameBarSuppressor(QObject* parent)
    : QObject(parent)
{
    m_packageFamilies = resolvePackageFamilies();

    m_killTimer.setInterval(300); // 0.3s: 押下即起動→即Kill を狙う（必要なら調整）
    m_killTimer.setSingleShot(false);
    connect(&m_killTimer, &QTimer::timeout, this, [this]{
        if (!m_enabled) return;
        killOnce();
    });
}

GameBarSuppressor::~GameBarSuppressor() {
    setSuppressed(false);
}

void GameBarSuppressor::setSuppressed(bool enabled)
{
    if (enabled == m_enabled) return;
    m_enabled = enabled;

    if (m_enabled) {
        apply();
        killOnce();
        m_killTimer.start();
    } else {
        m_killTimer.stop();
        restore();
    }
}

void GameBarSuppressor::apply()
{
    // 1) GameDVR/GameBar 系 (HKCU)
    readDword(HKEY_CURRENT_USER, L"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\GameDVR", L"AppCaptureEnabled", m_bk_AppCaptureEnabled);
    writeDword(HKEY_CURRENT_USER, L"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\GameDVR", L"AppCaptureEnabled", 0);

    readDword(HKEY_CURRENT_USER, L"System\\GameConfigStore", L"GameDVR_Enabled", m_bk_GameDVR_Enabled);
    writeDword(HKEY_CURRENT_USER, L"System\\GameConfigStore", L"GameDVR_Enabled", 0);

    readDword(HKEY_CURRENT_USER, L"SOFTWARE\\Microsoft\\GameBar", L"UseNexusForGameBarEnabled", m_bk_UseNexusForGameBarEnabled);
    writeDword(HKEY_CURRENT_USER, L"SOFTWARE\\Microsoft\\GameBar", L"UseNexusForGameBarEnabled", 0);

    // 2) BackgroundAccessApplications: PFNごとに Disabled/DisabledByUser=1
    m_bgBackups.clear();
    for (const auto& pfn : m_packageFamilies) {
        BgBackup b{};
        b.pfn = pfn;
        const std::wstring k = bgKeyPath(pfn);
        readDword(HKEY_CURRENT_USER, k.c_str(), L"Disabled", b.disabled);
        readDword(HKEY_CURRENT_USER, k.c_str(), L"DisabledByUser", b.disabledByUser);
        writeDword(HKEY_CURRENT_USER, k.c_str(), L"Disabled", 1);
        writeDword(HKEY_CURRENT_USER, k.c_str(), L"DisabledByUser", 1);
        m_bgBackups.push_back(std::move(b));
    }

    DebugLog(L"[GameBarSuppressor] Applied (focus ON). PFN count=" + std::to_wstring(m_packageFamilies.size()));
}

void GameBarSuppressor::restore()
{
    auto restoreOrDelete = [](const wchar_t* subKey, const wchar_t* name, const DwordBackup& bk) {
        if (bk.existed) writeDword(HKEY_CURRENT_USER, subKey, name, bk.value);
        else deleteValue(HKEY_CURRENT_USER, subKey, name);
    };

    restoreOrDelete(L"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\GameDVR", L"AppCaptureEnabled", m_bk_AppCaptureEnabled);
    restoreOrDelete(L"System\\GameConfigStore", L"GameDVR_Enabled", m_bk_GameDVR_Enabled);
    restoreOrDelete(L"SOFTWARE\\Microsoft\\GameBar", L"UseNexusForGameBarEnabled", m_bk_UseNexusForGameBarEnabled);

    for (const auto& b : m_bgBackups) {
        const std::wstring k = bgKeyPath(b.pfn);
        if (b.disabled.existed) writeDword(HKEY_CURRENT_USER, k.c_str(), L"Disabled", b.disabled.value);
        else deleteValue(HKEY_CURRENT_USER, k.c_str(), L"Disabled");

        if (b.disabledByUser.existed) writeDword(HKEY_CURRENT_USER, k.c_str(), L"DisabledByUser", b.disabledByUser.value);
        else deleteValue(HKEY_CURRENT_USER, k.c_str(), L"DisabledByUser");
    }
    m_bgBackups.clear();

    DebugLog(L"[GameBarSuppressor] Restored (focus OFF).");
}

void GameBarSuppressor::killOnce()
{
    // PFN一致Kill（exe名変化に強い）
    killByPackageFamily(m_packageFamilies);
    // 保険：exe名Kill（PFN取れないWin32側コンポーネント用）
    killByExeNames();
}

void GameBarSuppressor::killByPackageFamily(const std::vector<std::wstring>& pfns)
{
    if (pfns.empty()) return;

    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE) return;

    PROCESSENTRY32W pe{};
    pe.dwSize = sizeof(pe);
    if (!Process32FirstW(snap, &pe)) { CloseHandle(snap); return; }

    do {
        HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE, FALSE, pe.th32ProcessID);
        if (!h) continue;

        UINT32 len = 0;
        LONG rc = GetPackageFamilyName(h, &len, nullptr);
        if (rc == ERROR_INSUFFICIENT_BUFFER && len > 0) {
            std::wstring pfn;
            pfn.resize(len);
            rc = GetPackageFamilyName(h, &len, &pfn[0]);
            if (rc == ERROR_SUCCESS) {
                // 末尾の\0調整
                while (!pfn.empty() && pfn.back() == L'\0') pfn.pop_back();
                for (const auto& target : pfns) {
                    if (ieq(pfn, target)) {
                        TerminateProcess(h, 0);
                        break;
                    }
                }
            }
        }
        CloseHandle(h);
    } while (Process32NextW(snap, &pe));

    CloseHandle(snap);
}

void GameBarSuppressor::killByExeNames()
{
    const std::vector<std::wstring> names = {
        L"GameBar.exe",
        L"GameBarFTServer.exe",
        L"GameBarPresenceWriter.exe",
        L"XboxGameBar.exe",
        L"XboxGamingOverlay.exe"
    };

    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE) return;

    PROCESSENTRY32W pe{};
    pe.dwSize = sizeof(pe);
    if (!Process32FirstW(snap, &pe)) { CloseHandle(snap); return; }

    do {
        std::wstring exe = pe.szExeFile;
        bool hit = false;
        for (const auto& n : names) {
            if (ieq(exe, n)) { hit = true; break; }
        }
        if (!hit) continue;

        HANDLE h = OpenProcess(PROCESS_TERMINATE, FALSE, pe.th32ProcessID);
        if (!h) continue;
        TerminateProcess(h, 0);
        CloseHandle(h);
    } while (Process32NextW(snap, &pe));

    CloseHandle(snap);
}

std::vector<std::wstring> GameBarSuppressor::resolvePackageFamilies()
{
    std::vector<std::wstring> out;

    // 0) 既知（あなたの環境）
    out.push_back(L"Microsoft.XboxGamingOverlay_8wekyb3d8bbwe");

    // 1) BackgroundAccessApplications を列挙して "XboxGamingOverlay" 等を含むPFNを拾う（環境差対策）
    HKEY hKey = nullptr;
    if (RegOpenKeyExW(HKEY_CURRENT_USER,
                      L"Software\\Microsoft\\Windows\\CurrentVersion\\BackgroundAccessApplications",
                      0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        DWORD idx = 0;
        wchar_t name[512];
        DWORD nameLen = 512;
        while (RegEnumKeyExW(hKey, idx, name, &nameLen, nullptr, nullptr, nullptr, nullptr) == ERROR_SUCCESS) {
            std::wstring s(name);
            if (containsI(s, L"XboxGamingOverlay") || containsI(s, L"GamingOverlay") || containsI(s, L"GameBar")) {
                out.push_back(s);
            }
            idx++;
            nameLen = 512;
        }
        RegCloseKey(hKey);
    }

    dedup(out);
    return out;
}

void GameBarSuppressor::dedup(std::vector<std::wstring>& v)
{
    std::vector<std::wstring> o;
    for (const auto& s : v) {
        bool exists=false;
        for (const auto& t : o) { if (ieq(s,t)) { exists=true; break; } }
        if (!exists) o.push_back(s);
    }
    v.swap(o);
}

bool GameBarSuppressor::readDword(HKEY root, const wchar_t* subKey, const wchar_t* valueName, DwordBackup& out)
{
    out = {};
    HKEY h = nullptr;
    if (RegOpenKeyExW(root, subKey, 0, KEY_READ, &h) != ERROR_SUCCESS) return false;
    DWORD type=0, data=0, cb=sizeof(DWORD);
    LSTATUS st = RegQueryValueExW(h, valueName, nullptr, &type, reinterpret_cast<LPBYTE>(&data), &cb);
    RegCloseKey(h);
    if (st == ERROR_SUCCESS && type == REG_DWORD) {
        out.existed = true;
        out.value = data;
        return true;
    }
    out.existed = false;
    return false;
}

bool GameBarSuppressor::writeDword(HKEY root, const wchar_t* subKey, const wchar_t* valueName, DWORD v)
{
    HKEY h = nullptr;
    DWORD disp=0;
    if (RegCreateKeyExW(root, subKey, 0, nullptr, 0, KEY_WRITE, nullptr, &h, &disp) != ERROR_SUCCESS) return false;
    LSTATUS st = RegSetValueExW(h, valueName, 0, REG_DWORD, reinterpret_cast<const BYTE*>(&v), sizeof(DWORD));
    RegCloseKey(h);
    return st == ERROR_SUCCESS;
}

bool GameBarSuppressor::deleteValue(HKEY root, const wchar_t* subKey, const wchar_t* valueName)
{
    HKEY h = nullptr;
    if (RegOpenKeyExW(root, subKey, 0, KEY_SET_VALUE, &h) != ERROR_SUCCESS) return false;
    LSTATUS st = RegDeleteValueW(h, valueName);
    RegCloseKey(h);
    return st == ERROR_SUCCESS;
}
