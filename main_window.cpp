#include "main_window.h"
#include "KeyboardSender.h"
#include "DebugLog.h"
#include "GameBarSuppressor.h"
#include <cstdint>
#include <QEvent>
#include <QCloseEvent>
#include <QApplication>
#include <QKeyEvent>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QTabBar>
#include <QSizePolicy>
#include <QTimer>
#include <QSignalBlocker>
#include <QRegularExpression>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <vector>
#include <windows.h>
#include <hidusage.h>
#include <dbt.h>
#include <setupapi.h>
#include <devpropdef.h>
#include <cfgmgr32.h>
#include <hidsdi.h>
#include <hidpi.h>
#include <objbase.h>
#include <cwchar>
#include "AppShutdown.h"
#include "window.h"

// SDK互換：RIDEV_NOHOTKEYS が未定義なら定義（値は WinUser.h 互換）
#ifndef RIDEV_NOHOTKEYS
#define RIDEV_NOHOTKEYS 0x0200
#endif

// Global/Static pointer for hook callback
static MainWindow* g_mainWindow = nullptr;

// 半角/全角キーのオートリピート等で Alt+` を連打しないための簡易状態
static std::atomic<bool> g_hankakuHeld{ false };
static std::atomic<ULONGLONG> g_hankakuLastDownTick{ 0 };

// --- Winキー抑止（クライアントがアクティブな間はローカルOSのスタートメニュー等を開かせない） ---
// Raw Input (WM_INPUT) だけでは OS の Winキー処理を抑止できないため、WH_KEYBOARD_LL で握りつぶす。
// ※低レベルフックはデバイス識別ができないため、複数キーボード環境では「選択デバイスのみ抑止」は難しい。
static std::atomic<bool> g_remoteInputActive{ false };
static HHOOK g_llKeyboardHook = nullptr;
// Winキーを「Downを捕まえたらUpまで」確実に握るための捕捉状態
// bit0 = LWIN, bit1 = RWIN
static std::atomic<uint8_t> g_winCapturedMask{ 0 };
// Alt の押下状態（GetAsyncKeyState 依存をやめる）
static std::atomic<bool> g_altDown{ false };
// Win+Alt+B を検出して処理中（ローカルへ絶対に流さない）
static std::atomic<bool> g_winAltBActive{ false };

struct IgnoreEv {
    uint16_t make;
    uint16_t rawFlags; // RI_KEY_*（WM_INPUTのk.Flagsと同系）
    uint16_t vkey;
};
static std::mutex g_ignoreMtx;
static std::vector<IgnoreEv> g_ignoreOnce;
static std::atomic<ULONGLONG> g_ignoreExpire{ 0 };

static void SetIgnoreOnce(std::initializer_list<IgnoreEv> evs, ULONGLONG ttlMs = 200)
{
    std::lock_guard<std::mutex> lock(g_ignoreMtx);
    g_ignoreOnce.assign(evs.begin(), evs.end());
    g_ignoreExpire.store(GetTickCount64() + ttlMs, std::memory_order_relaxed);
}

static bool ConsumeIgnoreOnce(uint16_t make, uint16_t rawFlags, uint16_t vkey)
{
    const ULONGLONG now = GetTickCount64();
    if (now > g_ignoreExpire.load(std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> lock(g_ignoreMtx);
        g_ignoreOnce.clear();
        return false;
    }
    std::lock_guard<std::mutex> lock(g_ignoreMtx);
    for (auto it = g_ignoreOnce.begin(); it != g_ignoreOnce.end(); ++it) {
        if (it->make == make && it->rawFlags == rawFlags && it->vkey == vkey) {
            g_ignoreOnce.erase(it);
            return true;
        }
    }
    return false;
}

static void ForwardWinAltBToRemote(uint8_t winMask)
{
    // 押していたWinを反映（RWin優先、なければLWin）
    const bool useR = (winMask & 0x02) != 0;
    const uint16_t winMake = useR ? 0x5C : 0x5B;
    const uint16_t winVk   = useR ? VK_RWIN : VK_LWIN;

    const uint16_t altMake = 0x38;     // Left Alt
    const uint16_t altVk   = VK_MENU;
    const uint16_t bMake   = 0x30;     // 'B'
    const uint16_t bVk     = 'B';

    // WM_INPUT側が同じイベントを拾った場合に二重送信しないためのガード
    SetIgnoreOnce({
        { winMake, (uint16_t)(RI_KEY_E0),                winVk },
        { altMake, (uint16_t)0,                          altVk },
        { bMake,   (uint16_t)0,                          bVk   },
        { bMake,   (uint16_t)RI_KEY_BREAK,               bVk   },
        { altMake, (uint16_t)RI_KEY_BREAK,               altVk },
        { winMake, (uint16_t)(RI_KEY_E0 | RI_KEY_BREAK), winVk },
    });

    DebugLog(L"[WinAltB] Intercept locally -> forward synthetic Win+Alt+B to remote.");
    EnqueueKeyboardRawEvent(winMake, (uint16_t)RI_KEY_E0, winVk);
    EnqueueKeyboardRawEvent(altMake, (uint16_t)0, altVk);
    EnqueueKeyboardRawEvent(bMake,   (uint16_t)0, bVk);
    EnqueueKeyboardRawEvent(bMake,   (uint16_t)RI_KEY_BREAK, bVk);
    EnqueueKeyboardRawEvent(altMake, (uint16_t)RI_KEY_BREAK, altVk);
    EnqueueKeyboardRawEvent(winMake, (uint16_t)(RI_KEY_E0 | RI_KEY_BREAK), winVk);
}

static bool IsForegroundOurProcess()
{
    HWND fg = GetForegroundWindow();
    if (!fg) return false;
    DWORD pid = 0;
    GetWindowThreadProcessId(fg, &pid);
    return pid == GetCurrentProcessId();
}

static LRESULT CALLBACK LowLevelKeyboardProc(int nCode, WPARAM wParam, LPARAM lParam)
{
    const bool activeByForeground = IsForegroundOurProcess();
    const bool shouldHook = activeByForeground
        || g_remoteInputActive.load(std::memory_order_relaxed)
        || g_winCapturedMask.load(std::memory_order_relaxed);

    // アクティブ、もしくは「Winキー捕捉中(Up待ち)」であればフック処理を行う
    if (nCode == HC_ACTION && shouldHook) {
        const KBDLLHOOKSTRUCT* ks = reinterpret_cast<const KBDLLHOOKSTRUCT*>(lParam);
        if (ks) {
            const bool isUp = (wParam == WM_KEYUP || wParam == WM_SYSKEYUP) || ((ks->flags & LLKHF_UP) != 0);

            // Alt状態をフックイベントで追跡
            if (ks->vkCode == VK_LMENU || ks->vkCode == VK_RMENU) {
                g_altDown.store(!isUp, std::memory_order_relaxed);
            }

            // Winを捕捉中は、Win+Alt+B の Alt/B もローカルに流さない（Game Bar HDR トースト抑止）
            // ※WM_INPUT に BのKeyDownが来ないケースがあるため、ここで合成して送る
            const uint8_t winMask = g_winCapturedMask.load(std::memory_order_relaxed);
            const bool winCaptured = (winMask != 0);

            // Win+Alt+B の検出点：B KeyDown
            if (!isUp && winCaptured && g_altDown.load(std::memory_order_relaxed) && ks->vkCode == 'B') {
                g_winAltBActive.store(true, std::memory_order_relaxed);
                ForwardWinAltBToRemote(winMask);
                return 1; // ローカルへ流さない
            }

            // 合成送信中は Alt/B を確実にローカルへ流さない（B upで解除）
            if (g_winAltBActive.load(std::memory_order_relaxed)) {
                if (ks->vkCode == 'B' || ks->vkCode == VK_LMENU || ks->vkCode == VK_RMENU) {
                    if (ks->vkCode == 'B' && isUp) {
                        g_winAltBActive.store(false, std::memory_order_relaxed);
                    }
                    return 1;
                }
            }

            const bool isWin = (ks->vkCode == VK_LWIN || ks->vkCode == VK_RWIN);
            if (isWin) {
                const uint8_t mask = (ks->vkCode == VK_LWIN) ? 0x01 : 0x02;
                const uint16_t makeCode = (ks->vkCode == VK_LWIN) ? 0x5B : 0x5C;
                const uint16_t rawFlags = (uint16_t)(RI_KEY_E0 | (isUp ? RI_KEY_BREAK : 0));
                const uint16_t vkey = (uint16_t)ks->vkCode;

                if (!isUp) {
                    // Win Down: 捕捉開始 + ローカル抑止
                    // アクティブ時のみ捕捉開始
                    if (activeByForeground || g_remoteInputActive.load(std::memory_order_relaxed)) {
                        g_winCapturedMask.fetch_or(mask, std::memory_order_relaxed);
                        return 1; // ローカルへ伝播させない
                    }
                } else {
                    // Win Up: ローカル抑止（捕捉解除は必ず行う）
                    g_winCapturedMask.fetch_and((uint8_t)~mask, std::memory_order_relaxed);
                    return 1;
                }
            }
        }
    }
    return CallNextHookEx(g_llKeyboardHook, nCode, wParam, lParam);
}

static void InstallLowLevelKeyboardHook()
{
    if (g_llKeyboardHook) return;
    HMODULE hMod = GetModuleHandleW(nullptr);
    g_llKeyboardHook = SetWindowsHookExW(WH_KEYBOARD_LL, LowLevelKeyboardProc, hMod, 0);
    if (!g_llKeyboardHook) {
        DebugLog(L"[WinKeyHook] SetWindowsHookExW failed. err=" + std::to_wstring(GetLastError()));
    } else {
        DebugLog(L"[WinKeyHook] Installed successfully.");
    }
}

static void UninstallLowLevelKeyboardHook()
{
    if (!g_llKeyboardHook) return;
    UnhookWindowsHookEx(g_llKeyboardHook);
    g_llKeyboardHook = nullptr;
}

namespace {
// --- Fix for LNK2001: DEVPKEY_Device_* unresolved ---
// 一部のWindows SDKでは devpkey.h の DEVPKEY_Device_* が extern 宣言のみになり、
// INITGUID の扱い次第で実体が生成されず LNK2001 になることがある。
// そのため、必要な DEVPROPKEY をこの翻訳単位で確実に定義して使用する。
// ContainerId: formatID 8C7ED206-3F8A-4827-B3AB-AE9E1FAEFC6C, propID 2
static const DEVPROPKEY kDevpkey_Device_ContainerId = {
    { 0x8c7ed206, 0x3f8a, 0x4827, { 0xb3, 0xab, 0xae, 0x9e, 0x1f, 0xae, 0xfc, 0x6c } },
    2
};
// HardwareIds: formatID A45C254E-DF1C-4EFD-8020-67D146A850E0, propID 3
static const DEVPROPKEY kDevpkey_Device_HardwareIds = {
    { 0xa45c254e, 0xdf1c, 0x4efd, { 0x80, 0x20, 0x67, 0xd1, 0x46, 0xa8, 0x50, 0xe0 } },
    3
};

static bool TryExtractVidPid(const QString& path, quint16& outVid, quint16& outPid) {
    // 例: \\?\HID#VID_046D&PID_C31C&MI_00#...#{...}
    static const QRegularExpression re(QStringLiteral("VID_([0-9A-Fa-f]{4}).*PID_([0-9A-Fa-f]{4})"));
    const QRegularExpressionMatch m = re.match(path);
    if (!m.hasMatch()) return false;
    bool ok1 = false, ok2 = false;
    const int vid = m.captured(1).toInt(&ok1, 16);
    const int pid = m.captured(2).toInt(&ok2, 16);
    if (!ok1 || !ok2) return false;
    outVid = static_cast<quint16>(vid & 0xFFFF);
    outPid = static_cast<quint16>(pid & 0xFFFF);
    return true;
}

static bool QueryIsJapaneseKeyboardRawDevice(HANDLE hDev)
{
    RID_DEVICE_INFO info{};
    info.cbSize = sizeof(info);
    UINT infoSize = sizeof(info);
    if (GetRawInputDeviceInfo(hDev, RIDI_DEVICEINFO, &info, &infoSize) == (UINT)-1) {
        return false;
    }
    if (info.dwType != RIM_TYPEKEYBOARD) return false;
    // RID_DEVICE_INFO_KEYBOARD.dwType: 0x7 == Japanese keyboard (Microsoft Learn)
    return (info.keyboard.dwType == 0x7);
}

static QString FixupForWin32DevicePath(QString path) {
    // RawInput の RIDI_DEVICENAME が "\\??\\" で始まる環境がある。
    // SetupAPI / CreateFile は通常 "\\\\?\\" 形式を期待するため補正する。
    const QString ntPrefix = QStringLiteral("\\\\??\\\\");
    if (path.startsWith(ntPrefix)) {
        path.remove(0, ntPrefix.size());
        path.prepend(QStringLiteral("\\\\?\\"));
    }
    return path;
}

static bool TryDeriveDeviceInstanceId(const QString& rawDeviceName, QString& outInstanceId) {
    // RawInput名から PnP インスタンスID相当を作る。
    QString s = FixupForWin32DevicePath(rawDeviceName);
    const QString win32Prefix = QStringLiteral("\\\\?\\");
    if (s.startsWith(win32Prefix)) {
        s = s.mid(win32Prefix.size());
    }
    static const QRegularExpression reTail(QStringLiteral(R"(\#\{[0-9A-Fa-f\-]{36}\}$)"));
    s.remove(reTail);
    s.replace('#', '\\');
    outInstanceId = s;
    return !outInstanceId.isEmpty();
}

static bool TryGetContainerIdFromDevNode(const QString& instanceId, QString& outKey) {
    DEVINST devInst = 0;
    std::wstring idW = instanceId.toStdWString();
    CONFIGRET cr = CM_Locate_DevNodeW(&devInst, idW.data(), CM_LOCATE_DEVNODE_NORMAL);
    if (cr != CR_SUCCESS) return false;

    DEVPROPTYPE propType = 0;
    GUID container{};
    ULONG size = sizeof(container);
    cr = CM_Get_DevNode_PropertyW(devInst, &kDevpkey_Device_ContainerId, &propType, reinterpret_cast<PBYTE>(&container), &size, 0);
    if (cr != CR_SUCCESS) return false;

    wchar_t guidBuf[64]{};
    if (StringFromGUID2(container, guidBuf, 64) <= 0) return false;
    outKey = QString::fromWCharArray(guidBuf).toUpper();
    return true;
}

static bool TryGetHardwareIdsFromDevNode(const QString& instanceId, QStringList& outIds) {
    outIds.clear();
    DEVINST devInst = 0;
    std::wstring idW = instanceId.toStdWString();
    CONFIGRET cr = CM_Locate_DevNodeW(&devInst, idW.data(), CM_LOCATE_DEVNODE_NORMAL);
    if (cr != CR_SUCCESS) return false;

    DEVPROPTYPE propType = 0;
    ULONG size = 0;
    cr = CM_Get_DevNode_PropertyW(devInst, &kDevpkey_Device_HardwareIds, &propType, nullptr, &size, 0);
    if (cr != CR_BUFFER_SMALL || size == 0) return false;
    if (propType != DEVPROP_TYPE_STRING_LIST) return false;

    std::vector<wchar_t> buf(size / sizeof(wchar_t));
    cr = CM_Get_DevNode_PropertyW(devInst, &kDevpkey_Device_HardwareIds, &propType, reinterpret_cast<PBYTE>(buf.data()), &size, 0);
    if (cr != CR_SUCCESS) return false;

    const wchar_t* p = buf.data();
    while (*p) {
        outIds.push_back(QString::fromWCharArray(p));
        p += wcslen(p) + 1;
    }
    return !outIds.isEmpty();
}

static QString NormalizeDevicePathForKey(QString path) {
    path = FixupForWin32DevicePath(path);
    static const QRegularExpression reTail(QStringLiteral(R"(\#\{[0-9A-Fa-f\-]{36}\}$)"));
    path.remove(reTail);
    return path.toUpper();
}

static bool TryGetContainerIdKey(const QString& deviceInterfacePath, QString& outKey) {
    const QString fixedPath = FixupForWin32DevicePath(deviceInterfacePath);
    const std::wstring pathW = fixedPath.toStdWString();

    HDEVINFO devInfo = SetupDiCreateDeviceInfoList(nullptr, nullptr);
    if (devInfo == INVALID_HANDLE_VALUE) return false;

    SP_DEVICE_INTERFACE_DATA ifData{};
    ifData.cbSize = sizeof(ifData);

    if (!SetupDiOpenDeviceInterfaceW(devInfo, pathW.c_str(), 0, &ifData)) {
        SetupDiDestroyDeviceInfoList(devInfo);
        QString instanceId;
        if (TryDeriveDeviceInstanceId(deviceInterfacePath, instanceId)) {
            return TryGetContainerIdFromDevNode(instanceId, outKey);
        }
        return false;
    }

    SP_DEVINFO_DATA devData{};
    devData.cbSize = sizeof(devData);
    if (!SetupDiGetDeviceInterfaceDetailW(devInfo, &ifData, nullptr, 0, nullptr, &devData)) {
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
            SetupDiDestroyDeviceInfoList(devInfo);
            return false;
        }
    }

    DEVPROPTYPE propType = 0;
    GUID containerId{};
    if (!SetupDiGetDevicePropertyW(devInfo, &devData, &kDevpkey_Device_ContainerId, &propType, (PBYTE)&containerId, sizeof(containerId), nullptr, 0)) {
        SetupDiDestroyDeviceInfoList(devInfo);
        return false;
    }

    SetupDiDestroyDeviceInfoList(devInfo);
    wchar_t guidBuf[64]{};
    if (StringFromGUID2(containerId, guidBuf, 64) <= 0) return false;
    outKey = QString::fromWCharArray(guidBuf).toUpper();
    return true;
}

static bool IsHidKeyboardByPreparsedData(const QString& deviceInterfacePath) {
    const QString fixedPath = FixupForWin32DevicePath(deviceInterfacePath);
    const std::wstring pathW = fixedPath.toStdWString();

    HANDLE h = CreateFileW(pathW.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        h = CreateFileW(pathW.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr);
    }
    if (h == INVALID_HANDLE_VALUE) return false;

    PHIDP_PREPARSED_DATA ppd = nullptr;
    if (!HidD_GetPreparsedData(h, &ppd)) {
        CloseHandle(h);
        return false;
    }

    HIDP_CAPS caps{};
    bool isKeyboard = false;
    if (HidP_GetCaps(ppd, &caps) == HIDP_STATUS_SUCCESS) {
        if (caps.UsagePage == HID_USAGE_PAGE_GENERIC &&
            (caps.Usage == HID_USAGE_GENERIC_KEYBOARD || caps.Usage == HID_USAGE_GENERIC_KEYPAD)) {
            isKeyboard = true;
        }
    }

    HidD_FreePreparsedData(ppd);
    CloseHandle(h);
    return isKeyboard;
}

static QString Hex4(quint16 v) {
    return QStringLiteral("%1").arg(v, 4, 16, QChar('0')).toUpper();
}

static bool IsVirtualKeyboardDevicePath(const QString& path) {
    // 要件: 仮想キーボードは除外、内蔵は含める
    // 実用優先のヒューリスティック（まずRDPを確実に落とす）
    static const char* kTokens[] = {
        "RDP_KBD", "VMBUS", "HYPERV", "VIRTUAL", "VMWARE", "VBOX"
    };
    for (const char* t : kTokens) {
        if (path.contains(QLatin1String(t), Qt::CaseInsensitive)) return true;
    }
    return false;
}

static MainWindow::BusType DetectBusType(const QString& deviceInstanceId) {
    if (deviceInstanceId.isEmpty()) return MainWindow::BusType::Other;

    // 先に自己IDで判定できるものは確定してしまう（親探索のUSB早期return問題を潰す）
    const QString selfId = deviceInstanceId.toUpper();
    auto isBluetoothId = [](const QString& id) -> bool {
        return id.startsWith(QLatin1String("BTHENUM\\"))
            || id.startsWith(QLatin1String("BTHLEENUM\\"))
            || id.startsWith(QLatin1String("BTHLEDEVICE\\"))
            || id.startsWith(QLatin1String("BTH\\"))
            || id.startsWith(QLatin1String("BTHPORT\\"));
    };
    auto isUsbId = [](const QString& id) -> bool {
        return id.startsWith(QLatin1String("USB\\"));
    };
    auto isBuiltInId = [](const QString& id) -> bool {
        return id.startsWith(QLatin1String("ACPI\\"))
            || id.startsWith(QLatin1String("PCI\\"))
            || id.contains(QLatin1String("I8042PRT"));
    };

    if (isBluetoothId(selfId)) return MainWindow::BusType::Bluetooth;
    if (isBuiltInId(selfId))   return MainWindow::BusType::BuiltIn;
    if (isUsbId(selfId))       return MainWindow::BusType::USB;

    DEVINST devInst = 0;
    std::wstring idW = deviceInstanceId.toStdWString();
    CONFIGRET cr = CM_Locate_DevNodeW(&devInst, idW.data(), CM_LOCATE_DEVNODE_NORMAL);
    if (cr != CR_SUCCESS) return MainWindow::BusType::Other;

    // 親を辿ってバスを特定する
    DEVINST parent = 0;
    wchar_t buf[MAX_DEVICE_ID_LEN];

    DEVINST current = devInst;
    bool foundBluetooth = false;
    bool foundUsb = false;
    bool foundBuiltIn = false;

    // 最大10階層まで辿る（無限ループ防止）
    for (int depth = 0; depth < 10 && CM_Get_Parent(&parent, current, 0) == CR_SUCCESS; ++depth) {
        if (CM_Get_Device_IDW(parent, buf, MAX_DEVICE_ID_LEN, 0) == CR_SUCCESS) {
            QString parentId = QString::fromWCharArray(buf).toUpper();
            if (isBluetoothId(parentId)) foundBluetooth = true;
            if (isUsbId(parentId))       foundUsb = true;
            if (isBuiltInId(parentId))   foundBuiltIn = true;
        }
        current = parent;
    }

    // 優先順位: Bluetooth > USB > BuiltIn > Other
    if (foundBluetooth) return MainWindow::BusType::Bluetooth;
    if (foundUsb)       return MainWindow::BusType::USB;
    if (foundBuiltIn)   return MainWindow::BusType::BuiltIn;

    return MainWindow::BusType::Other;
}
} // namespace

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    ui.setupUi(this);
    g_mainWindow = this;
    if (qApp) qApp->installEventFilter(this);
    // Winキー抑止フックをインストール（ウィンドウがアクティブな間だけ有効化する）
    InstallLowLevelKeyboardHook();
    g_remoteInputActive.store(this->isActiveWindow(), std::memory_order_relaxed);

    // 初期値サイズ 1504*846 (16:9 領域 1280*720 を確保)
    // 横: 1280 + 224 = 1504, 縦: 720 + 126 = 846
    resize(1504, 846);
    setMinimumSize(1504, 846);

    // centralwidgetにレイアウトを追加して各ウィジェットを適切に配置
    if (ui.centralwidget) {
        QHBoxLayout* mainLayout = new QHBoxLayout(ui.centralwidget);
        mainLayout->setContentsMargins(10, 10, 10, 10);
        mainLayout->setSpacing(5);

        // 左側にタブウィジェットを追加
        if (ui.tabWidget) {
            ui.tabWidget->setDocumentMode(true);
            // タブバーとページ領域の境界線（QTabBarのベースライン）を描画しない
            // QTabWidget::tabBar() は protected のため findChild で取得する
            if (auto *bar = ui.tabWidget->findChild<QTabBar*>()) {
                bar->setDrawBase(false);
            }
            mainLayout->addWidget(ui.tabWidget);
        }

        // 右側に操作パネル用のレイアウトを作成
        QVBoxLayout* sideLayout = new QVBoxLayout();
        sideLayout->setSpacing(10);

        if (ui.groupBox) {
            ui.groupBox->setFixedWidth(159);
            // 重要: groupBox配下が"絶対配置"(geometry)のみだと、外側レイアウトに組み込んだ瞬間
            //       groupBoxのsizeHintが極小になり、タイトル行だけの高さに潰れて中のRadioButtonが
            //       クリップされることがある。
            //       → groupBox内部にもレイアウトを付け、sizeHint/最小高さがRadioButtonを含むようにする。
            if (ui.groupBox->layout() == nullptr) {
                auto* gbLayout = new QVBoxLayout(ui.groupBox);
                // タイトル領域の分だけ上マージンを多めに取る
                gbLayout->setContentsMargins(10, 24, 10, 10);
                gbLayout->setSpacing(6);

                auto addIf = [&](QWidget* w) {
                    if (w) gbLayout->addWidget(w);
                };
                addIf(ui.radioButton);
                addIf(ui.radioButton_2);
                addIf(ui.radioButton_3);
                addIf(ui.radioButton_4);

                // groupBox自体が縮み過ぎないよう、RadioButtonのsizeHintから最小高さを計算して設定
                int visibleCount = 0;
                int minH = gbLayout->contentsMargins().top() + gbLayout->contentsMargins().bottom();
                for (auto* rb : {ui.radioButton, ui.radioButton_2, ui.radioButton_3, ui.radioButton_4}) {
                    if (!rb) continue;
                    ++visibleCount;
                    minH += rb->sizeHint().height();
                }
                if (visibleCount >= 2) {
                    minH += gbLayout->spacing() * (visibleCount - 1);
                }
                // フレーム/スタイルの誤差吸収
                minH += 8;

                // Designer上の想定サイズ(高さ161)を下回らないようにする
                if (minH < 161) {
                    minH = 161;
                }

                ui.groupBox->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Minimum);
                ui.groupBox->setMinimumHeight(minH);
            }

            // 上寄せで配置（間にstretchが入るため、意図した高さを維持したまま上に貼り付く）
            sideLayout->addWidget(ui.groupBox, 0, Qt::AlignTop);
        }

        sideLayout->addStretch();

        if (ui.label) {
            sideLayout->addWidget(ui.label);
        }
        if (ui.comboBox) {
            sideLayout->addWidget(ui.comboBox);
        }

        mainLayout->addLayout(sideLayout);
        mainLayout->setStretch(0, 1); // tabWidgetを伸縮させる
        mainLayout->setStretch(1, 0); // 操作パネルは固定幅に近い扱い
    }

    // タブ内にレイアウトを設定してRenderHostWidgetsをアスペクト比維持で配置
    if (ui.tab && ui.frame) {
        QGridLayout* tabLayout = new QGridLayout(ui.tab);
        tabLayout->setContentsMargins(20, 20, 20, 20); // 上下左右20pxの余白を設定
        // RenderHostWidgetsはheightForWidthを持つため、レイアウト内で16:9が維持されるように配置
        tabLayout->addWidget(ui.frame, 0, 0);
    }

    // --- Keyboard ComboBox init / hotplug ---
    registerKeyboardDeviceNotifications();


    refreshKeyboardList();

    // selection cache
    if (ui.comboBox) {
        m_selectedKeyboardPath = ui.comboBox->currentData().toString();
        m_selectedKeyboardUniqueKey = ui.comboBox->currentData(Qt::UserRole + 2).toString();
        QObject::connect(ui.comboBox, QOverload<int>::of(&QComboBox::activated),
            this, [this](int index) {
                if (!ui.comboBox || index < 0) return;
                m_selectedKeyboardPath = ui.comboBox->currentData().toString();
                m_selectedKeyboardUniqueKey = ui.comboBox->currentData(Qt::UserRole + 2).toString();
            });
    }

    // initial focus state
    EnqueueKeyboardFocusChanged(this->isActiveWindow());

    // フォーカス中だけGame Barを止める（要件B）
    m_gameBarSuppressor = new GameBarSuppressor(this);
}

MainWindow::~MainWindow() {
    if (qApp) qApp->removeEventFilter(this);
    if (m_gameBarSuppressor) {
        m_gameBarSuppressor->setSuppressed(false); // 念のため復帰
    }
    UninstallLowLevelKeyboardHook();
    if (g_mainWindow == this) g_mainWindow = nullptr;
}

RenderHostWidgets* MainWindow::getRenderFrame() const {
    return ui.frame;
}

void MainWindow::closeEvent(QCloseEvent *event) {
    RequestShutdown();
    QMainWindow::closeEvent(event);
}

bool MainWindow::event(QEvent* e)
{
    if (e->type() == QEvent::WindowActivate) {
        g_remoteInputActive.store(true, std::memory_order_relaxed);
        EnqueueKeyboardFocusChanged(true);
        registerKeyboardDeviceNotifications();

        if (m_gameBarSuppressor) m_gameBarSuppressor->setSuppressed(true);
    } else if (e->type() == QEvent::WindowDeactivate) {
        g_remoteInputActive.store(false, std::memory_order_relaxed);
        // 半角/全角の押下状態が key-up 未検知で残ると、以後ずっと送信されない可能性があるため
        // 非アクティブ化時に念のため状態をクリアしておく。
        g_hankakuHeld.store(false, std::memory_order_relaxed);

        EnqueueKeyboardFocusChanged(false);
        registerKeyboardDeviceNotifications();

        if (m_gameBarSuppressor) m_gameBarSuppressor->setSuppressed(false);
    }
    return QMainWindow::event(e);
}

void MainWindow::resizeEvent(QResizeEvent *event) {
    // 再帰呼び出しを防ぐためのガード
    static bool inResize = false;
    if (inResize) return;
    inResize = true;

    // 内部レイアウトの固定オーバーヘッドを差し引いてレンダリング領域を計算
    // 横: 余白(10+10) + タブ内余白(20+20=40) + 間隔(5) + サイドパネル(159) = 224
    // 縦: 余白(10+10) + タブ内余白(20+20=40) + メニューバー・タブバー等(~66) = 126
    const int oh_w = 224;
    const int oh_h = 126;

    int targetW = width() - oh_w;
    int targetH = height() - oh_h;

    int snappedW, snappedH;
    SnapToKnownResolution(targetW, targetH, snappedW, snappedH);

    int nextW = snappedW + oh_w;
    int nextH = snappedH + oh_h;

    if (width() != nextW || height() != nextH) {
        resize(nextW, nextH);
        inResize = false;
        return;
    }

    QMainWindow::resizeEvent(event);
    inResize = false;

    // リサイズ中も描画を継続するためにRenderFrameを呼び出す
    // window.cpp側で非ブロッキング化されているため安全に呼び出せる
    RenderFrame();
}

bool MainWindow::nativeEvent(const QByteArray &eventType, void *message, qintptr *result) {
    if (eventType == "windows_generic_MSG") {
        MSG *msg = static_cast<MSG *>(message);
        if (msg->message == WM_NCHITTEST) {
            // デフォルトの挙動でどこがヒットしたか取得
            *result = DefWindowProc(msg->hwnd, msg->message, msg->wParam, msg->lParam);
            // 右下のリサイズハンドル(HTBOTTOMRIGHT)以外のリサイズハンドルを制限
            // 他の辺や角のリサイズハンドルが検出された場合、単なる境界(HTBORDER)として扱いリサイズを無効化する
            if (*result == HTLEFT || *result == HTRIGHT || *result == HTTOP ||
                *result == HTTOPLEFT || *result == HTTOPRIGHT ||
                *result == HTBOTTOM || *result == HTBOTTOMLEFT) {
                *result = HTBORDER;
                return true;
            }
        } else if (msg->message == WM_SIZING) {
            // 右下のリサイズのみを許可している前提で、アスペクト比16:9を維持
            RECT *rect = reinterpret_cast<RECT *>(msg->lParam);
            int w = rect->right - rect->left;

            // 幅に基づいて高さを16:9に調整
            int targetH = w * 9 / 16;

            // 最小サイズ(1504x846)を維持
            if (w < 1504) {
                w = 1504;
                targetH = 846;
                rect->right = rect->left + w;
            }

            rect->bottom = rect->top + targetH;

            *result = TRUE;
            return true;
        } else if (msg->message == WM_INPUT_DEVICE_CHANGE) {
            // Raw Input device hotplug notification (arrival/removal)
            // wParam: GIDC_ARRIVAL / GIDC_REMOVAL, lParam: HANDLE of device
            scheduleKeyboardListRefresh();
            *result = 0;
            return false;
        } else if (msg->message == WM_DEVICECHANGE) {
            // 環境差フォールバック（重い処理はデバウンス後に行う）
            scheduleKeyboardListRefresh();
            *result = 0;
            return false;
        } else if (msg->message == WM_INPUT) {
            if (m_selectedKeyboardUniqueKey.isEmpty()) {
                *result = 0;
                return false;
            }

            HRAWINPUT hRaw = (HRAWINPUT)msg->lParam;

            // 1. まずヘッダーだけ取得してデバイス判定を先に行う (効率化)
            RAWINPUTHEADER header;
            UINT headerSize = sizeof(RAWINPUTHEADER);
            if (GetRawInputData(hRaw, RID_HEADER, &header, &headerSize, sizeof(RAWINPUTHEADER)) == (UINT)-1) {
                *result = 0;
                return false;
            }

            if (header.dwType != RIM_TYPEKEYBOARD) {
                *result = 0;
                return false;
            }

            QString uniqueKey;
            auto it = m_handleToUniqueKey.find(header.hDevice);
            if (it != m_handleToUniqueKey.end()) {
                uniqueKey = it.value();
            } else {
                // 初見のハンドルの場合、パスから解決してキャッシュ
                UINT nameChars = 0;
                if (GetRawInputDeviceInfo(header.hDevice, RIDI_DEVICENAME, nullptr, &nameChars) != (UINT)-1 && nameChars > 0) {
                    std::wstring name(nameChars, L'\0');
                    if (GetRawInputDeviceInfo(header.hDevice, RIDI_DEVICENAME, name.data(), &nameChars) != (UINT)-1) {
                        if (!name.empty() && name.back() == L'\0') name.pop_back();
                        QString path = QString::fromWCharArray(name.c_str());
                        if (!TryGetContainerIdKey(path, uniqueKey)) {
                            uniqueKey = NormalizeDevicePathForKey(path);
                        }
                        m_handleToUniqueKey[header.hDevice] = uniqueKey;
                    }
                }
            }

            if (uniqueKey != m_selectedKeyboardUniqueKey) {
                // 選択外のデバイスは無視
                // 任意：調査用。頻繁に出るとログが膨らむので必要なら条件付きにする
                DebugLog(L"[WM_INPUT][DROP] device mismatch. got=" +
                         uniqueKey.toStdWString() + L" sel=" + m_selectedKeyboardUniqueKey.toStdWString());
                *result = 0;
                return false;
            }

            // Selected device is confirmed. Determine whether this raw keyboard device is JP(106/109).
            bool isJapaneseKeyboard = false;
            auto itJ = m_handleToIsJapanese.find(header.hDevice);
            if (itJ != m_handleToIsJapanese.end()) {
                isJapaneseKeyboard = itJ.value();
            } else {
                isJapaneseKeyboard = QueryIsJapaneseKeyboardRawDevice(header.hDevice);
                m_handleToIsJapanese[header.hDevice] = isJapaneseKeyboard;
            }

            // 2. 選択デバイスと一致した場合のみ、全データを取得
            UINT size = 0;
            GetRawInputData(hRaw, RID_INPUT, nullptr, &size, sizeof(RAWINPUTHEADER));
            if (size == 0) {
                *result = 0;
                return false;
            }

            std::vector<uint8_t> buf(size);
            if (GetRawInputData(hRaw, RID_INPUT, buf.data(), &size, sizeof(RAWINPUTHEADER)) != size) {
                *result = 0;
                return false;
            }

            const RAWINPUT* ri = (const RAWINPUT*)buf.data();
            const RAWKEYBOARD& k = ri->data.keyboard;

            // 重要:
            //   RIDEV_INPUTSINK を使っているため、非アクティブでもWM_INPUTが来る。
            //   ここで非アクティブ即returnすると、Winキー等でフォーカスが外れた後の「KEY UP」が捨てられ、
            //   リモート側が押しっぱなしになる。
            //   → 非アクティブ時は「UPだけ通す」(Downは無視)。
            //   ただし、Winキー(0x5B/0x5C)に関しては、ローカルStartメニューでフォーカスが奪われた場合でも
            //   リモートへ送信したいため、例外的にDownも通す(保険)。
            const bool activeNow = this->isActiveWindow();
            const bool isUpNow = (k.Flags & RI_KEY_BREAK) != 0;
            const bool isWinSc = (k.MakeCode == 0x5B || k.MakeCode == 0x5C);
            const bool winCaptured = (g_winCapturedMask.load(std::memory_order_relaxed) != 0);

            // 非アクティブ時のDown破棄は基本維持。ただし：
            //  1) Winキー(sc=0x5B/0x5C) は例外でDownも通す（コメント通りの保険）
            //  2) すでにWin捕捉中は、Chord成立に必要なDown(Alt/B等)を落とさない
            if (!activeNow && !isUpNow && !isWinSc && !winCaptured) {
                DebugLog(L"[WM_INPUT][DROP] inactive down ignored. make=0x" +
                         std::to_wstring(k.MakeCode) +
                         L" flags=0x" + std::to_wstring((uint16_t)k.Flags));
                *result = 0;
                return false;
            }

            // MakeCode + Flags を送る（文字変換はしない）
            // ※RAWINPUTのVKeyは 0 / 0x00FF のことがあるため、必要ならレイアウトから補完する
            uint16_t vkRaw = (uint16_t)k.VKey;

            // (診断用) Winキーのログ
            if (vkRaw == VK_LWIN || vkRaw == VK_RWIN || k.MakeCode == 0x5B || k.MakeCode == 0x5C) {
                 DebugLog(L"[WM_INPUT] WinKey detected. MakeCode=0x" + std::to_wstring(k.MakeCode) +
                          L" Flags=0x" + std::to_wstring(k.Flags) +
                          L" VKeyRaw=0x" + std::to_wstring(vkRaw) +
                          L" Active=" + std::to_wstring(activeNow ? 1 : 0));
            }
            uint16_t vk = vkRaw;

            // Winキーはスキャンコードで確定できるので強制的に安定化
            if (k.MakeCode == 0x5B) vk = VK_LWIN;
            if (k.MakeCode == 0x5C) vk = VK_RWIN;

            // WM_INPUT側でもWin捕捉状態を更新して、Hook側の「Upまで握る」挙動を補強
            if (isWinSc) {
                const uint8_t mask = (k.MakeCode == 0x5B) ? 0x01 : 0x02;
                if (!isUpNow) g_winCapturedMask.fetch_or(mask, std::memory_order_relaxed);
                else          g_winCapturedMask.fetch_and((uint8_t)~mask, std::memory_order_relaxed);
            }

            if (vk == 0 || vk == 0x00FF) {
                // RI_KEY_E0/E1 を MapVirtualKeyExW に反映
                UINT sc = (UINT)k.MakeCode;
                if (k.Flags & RI_KEY_E0) sc |= 0xE000;
                if (k.Flags & RI_KEY_E1) sc |= 0xE100;
                const HKL hkl = GetKeyboardLayout(0);
                const UINT mapped = MapVirtualKeyExW(sc, MAPVK_VSC_TO_VK_EX, hkl);
                vk = (uint16_t)(mapped & 0xFFFF);
            }
            if (vk == 0x00FF) vk = 0;

            // (診断用) 0x29 は JIS では半角/全角、US では `(~) の位置。どのVKeyで来ているか確認する。
            if (k.MakeCode == 0x29) {
                DebugLog(L"[WM_INPUT] MakeCode=0x29 detected. rawVKey=0x" + std::to_wstring(vkRaw) +
                         L" resolvedVKey=0x" + std::to_wstring(vk) +
                         L" isJapaneseKeyboard=" + std::to_wstring(isJapaneseKeyboard ? 1 : 0) +
                         L" flags=0x" + std::to_wstring((uint16_t)k.Flags));
            }

            // 半角/全角: 選択デバイスからの入力だけを対象に、リモートへは Alt+` を送る
            // - JIS: 物理0x29が半角/全角 (典型VKey: VK_OEM_AUTO/VK_OEM_ENLW/VK_KANJI)
            // - US : 物理0x29が `(~) (VKey: VK_OEM_3) → 変換しない
            // Fix: HKL依存で vk が VK_OEM_3 に揺れても、デバイスが日本語キーボードなら 0x29 を半角/全角として扱う。
            const bool isHankakuZenkaku =
                (k.MakeCode == 0x29) &&
                (isJapaneseKeyboard || (vk == VK_OEM_AUTO || vk == VK_OEM_ENLW || vk == VK_KANJI));

            if (isHankakuZenkaku) {
                const bool isUp = (k.Flags & RI_KEY_BREAK) != 0;
                if (isUp) {
                    g_hankakuHeld.store(false, std::memory_order_relaxed);
                    *result = 0;
                    return false;
                }

                const ULONGLONG now = GetTickCount64();
                const bool wasHeld = g_hankakuHeld.exchange(true, std::memory_order_relaxed);
                const ULONGLONG last = g_hankakuLastDownTick.load(std::memory_order_relaxed);
                const bool stuck = (wasHeld && last != 0 && (now - last) > 1500);

                if (!wasHeld || stuck) {
                    if (stuck) {
                        DebugLog(L"[WM_INPUT][WARN] Hankaku held seems stuck. Resetting and sending Alt+`.");
                    }
                    g_hankakuLastDownTick.store(now, std::memory_order_relaxed);

                    // リモート側IMEトグル: Alt + ` を送る
                    const uint16_t altScan   = 0x38;      // Left Alt
                    const uint16_t altVk     = VK_MENU;   // 0x12
                    const uint16_t graveScan = 0x29;      // OEM_3 (`~) physical
                    const uint16_t graveVk   = VK_OEM_3;  // 0xC0

                    DebugLog(L"[WM_INPUT] Hankaku/Zenkaku -> send Alt+` to remote. "
                             L"Alt(scan=0x38,vk=0x12), Grave(scan=0x29,vk=0xC0)");

                    EnqueueKeyboardRawEvent(altScan,   RI_KEY_MAKE,  altVk);
                    EnqueueKeyboardRawEvent(graveScan, RI_KEY_MAKE,  graveVk);
                    EnqueueKeyboardRawEvent(graveScan, RI_KEY_BREAK, graveVk);
                    EnqueueKeyboardRawEvent(altScan,   RI_KEY_BREAK, altVk);
                } else {
                    DebugLog(L"[WM_INPUT] Hankaku ignored due to held(debounce).");
                }

                *result = 0;
                return false;
            }

            // 通常キー: MakeCode + Flags + (補完済み)VKey を送る
            if (vk == 0 || vk == 0x00FF) vk = 0;

            // フック側で Win+Alt+B を合成送信した直後、WM_INPUT に同じイベントが来たら二重送信なので捨てる
            if (ConsumeIgnoreOnce((uint16_t)k.MakeCode, (uint16_t)k.Flags, (uint16_t)vk)) {
                DebugLog(L"[WM_INPUT][SKIP] ignored (Win+Alt+B synthetic already sent).");
                *result = 0;
                return false;
            }

            // デバッグ：Win+ショートカットのため、全キーの生入力を追跡
            {
                const bool isUp = (k.Flags & RI_KEY_BREAK) != 0;
                DebugLog(
                    L"[WM_INPUT][KEY] make=0x" + std::to_wstring(k.MakeCode) +
                    L" flags=0x" + std::to_wstring((uint16_t)k.Flags) +
                    L" vkey=0x" + std::to_wstring((uint16_t)vk) +
                    L" up=" + std::to_wstring(isUp ? 1 : 0) +
                    L" active=" + std::to_wstring(activeNow ? 1 : 0)
                );
            }
            EnqueueKeyboardRawEvent((uint16_t)k.MakeCode, (uint16_t)k.Flags, vk);

            *result = 0;
            return false;
        }
    }
    return QMainWindow::nativeEvent(eventType, message, result);
}

void MainWindow::registerKeyboardDeviceNotifications() {
    // HWND生成を強制（Qtのネイティブハンドル）
    const HWND hwnd = reinterpret_cast<HWND>(winId());
    if (!hwnd) return;

    RAWINPUTDEVICE rid{};
    rid.usUsagePage = HID_USAGE_PAGE_GENERIC;          // 0x01
    rid.usUsage = HID_USAGE_GENERIC_KEYBOARD;         // 0x06
    // RIDEV_INPUTSINK: フォーカスがなくても入力を受ける（生入力は受け続ける）
    // RIDEV_DEVNOTIFY: デバイス変更通知を受ける
    // RIDEV_NOHOTKEYS: フォーカス中はWinホットキー(スタートメニュー等)をOS側で抑止する
    DWORD flags = RIDEV_DEVNOTIFY | RIDEV_INPUTSINK;
    if (this->isActiveWindow()) {
        flags |= RIDEV_NOHOTKEYS;
    }
    rid.dwFlags = flags;
    rid.hwndTarget = hwnd;

    if (!RegisterRawInputDevices(&rid, 1, sizeof(rid))) {
        DebugLog(L"[RawInput] RegisterRawInputDevices failed. err=" + std::to_wstring(GetLastError()));
    } else {
        DebugLog(L"[RawInput] RegisterRawInputDevices ok. flags=" + std::to_wstring((unsigned long)flags));
    }
}

void MainWindow::scheduleKeyboardListRefresh() {
    if (m_keyboardRefreshPending) return;
    m_keyboardRefreshPending = true;
    QTimer::singleShot(100, this, [this]() {
        m_keyboardRefreshPending = false;
        refreshKeyboardList();
    });
}

std::vector<MainWindow::KeyboardEntry> MainWindow::enumerateKeyboards() const {
    std::vector<KeyboardEntry> out;

    UINT deviceCount = 0;
    if (GetRawInputDeviceList(nullptr, &deviceCount, sizeof(RAWINPUTDEVICELIST)) != 0) {
        return out;
    }
    if (deviceCount == 0) return out;

    std::vector<RAWINPUTDEVICELIST> list(deviceCount);
    if (GetRawInputDeviceList(list.data(), &deviceCount, sizeof(RAWINPUTDEVICELIST)) == (UINT)-1) {
        return out;
    }

    out.reserve(deviceCount);

    for (UINT i = 0; i < deviceCount; ++i) {
        const HANDLE hDev = list[i].hDevice;
        const DWORD type = list[i].dwType;

        // device path を先に取得（判定に使うため）
        UINT nameChars = 0;
        if (GetRawInputDeviceInfo(hDev, RIDI_DEVICENAME, nullptr, &nameChars) == (UINT)-1 || nameChars == 0) {
            continue;
        }
        std::wstring name;
        name.resize(nameChars);
        if (GetRawInputDeviceInfo(hDev, RIDI_DEVICENAME, name.data(), &nameChars) == (UINT)-1) {
            continue;
        }
        if (!name.empty() && name.back() == L'\0') name.pop_back();
        const QString devicePath = QString::fromWCharArray(name.c_str());

        // 仮想キーボード除外
        if (IsVirtualKeyboardDevicePath(devicePath)) {
            continue;
        }

        RID_DEVICE_INFO info{};
        info.cbSize = sizeof(info);
        UINT infoSize = sizeof(info);
        if (GetRawInputDeviceInfo(hDev, RIDI_DEVICEINFO, &info, &infoSize) == (UINT)-1) {
            continue;
        }

        bool isKeyboard = false;
        if (type == RIM_TYPEKEYBOARD) {
            isKeyboard = true;
        } else if (type == RIM_TYPEHID) {
            if (info.hid.usUsagePage == HID_USAGE_PAGE_GENERIC &&
                (info.hid.usUsage == HID_USAGE_GENERIC_KEYBOARD || info.hid.usUsage == HID_USAGE_GENERIC_KEYPAD)) {
                isKeyboard = true;
            } else {
                // Usage が取れない/違う場合でも、CAPSから判定を試みる（Bluetooth等）
                if (IsHidKeyboardByPreparsedData(devicePath)) {
                    isKeyboard = true;
                }
            }
        }
        if (!isKeyboard) continue;

        KeyboardEntry e;
        e.devicePath = devicePath;

        // 重複排除のためのキー生成 (ContainerId優先)
        if (!TryGetContainerIdKey(devicePath, e.uniqueKey)) {
            e.uniqueKey = NormalizeDevicePathForKey(devicePath);
        }

        QString instanceId;
        TryDeriveDeviceInstanceId(devicePath, instanceId);
        e.busType = DetectBusType(instanceId);

        quint16 vid = 0xFFFF, pid = 0xFFFF;
        if (TryExtractVidPid(devicePath, vid, pid)) {
            e.vid = vid;
            e.pid = pid;
            e.hasVidPid = true;
        } else if (type == RIM_TYPEHID) {
            // フォールバック: HIDのVendor/Product
            e.vid = static_cast<quint16>(info.hid.dwVendorId & 0xFFFF);
            e.pid = static_cast<quint16>(info.hid.dwProductId & 0xFFFF);
            e.hasVidPid = true;
        } else {
            // RIM_TYPEKEYBOARD でも HardwareIds から狙う
            if (!instanceId.isEmpty()) {
                QStringList hwids;
                if (TryGetHardwareIdsFromDevNode(instanceId, hwids)) {
                    quint16 v2 = 0xFFFF, p2 = 0xFFFF;
                    for (const QString& h : hwids) {
                        if (TryExtractVidPid(h, v2, p2)) {
                            e.vid = v2;
                            e.pid = p2;
                            e.hasVidPid = true;
                            break;
                        }
                    }
                }
            }
            if (!e.hasVidPid) {
                e.vid = 0xFFFF;
                e.pid = 0xFFFF;
                e.hasVidPid = false;
            }
        }

        e.hDevice = hDev;
        out.push_back(std::move(e));
    }

    std::sort(out.begin(), out.end(), [](const KeyboardEntry& a, const KeyboardEntry& b) {
        if (a.busType != b.busType) return static_cast<int>(a.busType) < static_cast<int>(b.busType);
        if (a.vid != b.vid) return a.vid < b.vid;
        if (a.pid != b.pid) return a.pid < b.pid;
        return a.uniqueKey < b.uniqueKey;
    });

    // uniqueKey で重複排除
    out.erase(std::unique(out.begin(), out.end(), [](const KeyboardEntry& a, const KeyboardEntry& b) {
        return a.uniqueKey == b.uniqueKey;
    }), out.end());

    return out;
}

void MainWindow::refreshKeyboardList() {
    if (!ui.comboBox) return;

    const QString prevPath = ui.comboBox->currentData().toString();
    const auto keyboards = enumerateKeyboards();

    QSignalBlocker blocker(ui.comboBox);
    ui.comboBox->clear();

    int builtInIndex = 0;
    int usbIndex = 0;
    int btIndex = 0;
    int otherIndex = 0;

    for (int i = 0; i < static_cast<int>(keyboards.size()); ++i) {
        const auto& k = keyboards[i];

        const QString vidStr = k.hasVidPid ? Hex4(k.vid) : QStringLiteral("N/A");
        const QString pidStr = k.hasVidPid ? Hex4(k.pid) : QStringLiteral("N/A");

        QString prefix;
        int typeNo = 0;
        switch (k.busType) {
        case BusType::BuiltIn:
            prefix = QStringLiteral("Internal Keyboard");
            typeNo = ++builtInIndex;
            break;
        case BusType::USB:
            prefix = QStringLiteral("USB Keyboard");
            typeNo = ++usbIndex;
            break;
        case BusType::Bluetooth:
            prefix = QStringLiteral("BT Keyboard");
            typeNo = ++btIndex;
            break;
        default:
            prefix = QStringLiteral("Keyboard");
            typeNo = ++otherIndex;
            break;
        }

        // English labels: keep a space between prefix and index.
        const QString label = QStringLiteral("%1 %2 (VID:%3 PID:%4)")
            .arg(prefix)
            .arg(typeNo)
            .arg(vidStr)
            .arg(pidStr);

        // userData に devicePath を入れて選択維持に使う
        ui.comboBox->addItem(label, k.devicePath);
        ui.comboBox->setItemData(i, k.devicePath, Qt::ToolTipRole);
        // UserRole + 1 に HANDLE を隠し持つ (レガシー/互換用)
        ui.comboBox->setItemData(i, QVariant::fromValue((void*)k.hDevice), Qt::UserRole + 1);
        // UserRole + 2 に uniqueKey を持つ (堅牢なマッチング用)
        ui.comboBox->setItemData(i, k.uniqueKey, Qt::UserRole + 2);
    }

    // 更新ごとにハンドルマップをリセット
    m_handleToUniqueKey.clear();
    m_handleToIsJapanese.clear();
    for (const auto& k : keyboards) {
        m_handleToUniqueKey[k.hDevice] = k.uniqueKey;
    }

    // 選択維持
    if (!prevPath.isEmpty()) {
        for (int i = 0; i < ui.comboBox->count(); ++i) {
            if (ui.comboBox->itemData(i).toString() == prevPath) {
                ui.comboBox->setCurrentIndex(i);
                // Update internal state explicitly as activated() signal is not emitted programmatically
                m_selectedKeyboardPath = ui.comboBox->currentData().toString();
                m_selectedKeyboardUniqueKey = ui.comboBox->currentData(Qt::UserRole + 2).toString();
                return;
            }
        }
    }
    if (ui.comboBox->count() > 0) {
        ui.comboBox->setCurrentIndex(0);
        // Update internal state explicitly
        m_selectedKeyboardPath = ui.comboBox->currentData().toString();
        m_selectedKeyboardUniqueKey = ui.comboBox->currentData(Qt::UserRole + 2).toString();
    }
}

bool MainWindow::eventFilter(QObject* watched, QEvent* e)
{
    Q_UNUSED(watched);

    // ピンポイント：Win+Alt+Down だけローカルUIへ流さない
    if (e->type() == QEvent::KeyPress || e->type() == QEvent::KeyRelease || e->type() == QEvent::ShortcutOverride) {
        // Winはフックで握り潰されることがあるので、物理状態を GetAsyncKeyState で判定する。
        const SHORT winL = GetAsyncKeyState(VK_LWIN);
        const SHORT winR = GetAsyncKeyState(VK_RWIN);
        const SHORT alt  = GetAsyncKeyState(VK_MENU);
        const bool winHeld = (winL & 0x8000) || (winR & 0x8000);
        const bool altHeld = (alt  & 0x8000);

        if (winHeld && altHeld) {
            auto* ke = static_cast<QKeyEvent*>(e);
            if (ke && ke->key() == Qt::Key_Down) {
                // 保険：すでに開いていたら閉じる（開きかけも潰す）
                if (ui.comboBox) ui.comboBox->hidePopup();
                e->accept();
                return true;
            }
        }
    }
    return QMainWindow::eventFilter(watched, e);
}
