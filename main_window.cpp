#include "main_window.h"
#include "KeyboardSender.h"
#include <QEvent>
#include <QCloseEvent>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QTabBar>
#include <QSizePolicy>
#include <QTimer>
#include <QSignalBlocker>
#include <QRegularExpression>
#include <algorithm>
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
} // namespace

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    ui.setupUi(this);

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
        m_selectedKeyboardHandle = (HANDLE)ui.comboBox->currentData(Qt::UserRole + 1).value<void*>();
        QObject::connect(ui.comboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int index) {
                if (!ui.comboBox || index < 0) return;
                m_selectedKeyboardPath = ui.comboBox->currentData().toString();
                m_selectedKeyboardHandle = (HANDLE)ui.comboBox->currentData(Qt::UserRole + 1).value<void*>();
            });
    }

    // initial focus state
    EnqueueKeyboardFocusChanged(this->isActiveWindow());
}

MainWindow::~MainWindow() {}

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
        EnqueueKeyboardFocusChanged(true);
    } else if (e->type() == QEvent::WindowDeactivate) {
        EnqueueKeyboardFocusChanged(false);
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
            // フォーカス中のみ（Qt側でもactive制御しているが保険）
            if (!this->isActiveWindow()) {
                *result = 0;
                return false;
            }
            if (!m_selectedKeyboardHandle) {
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

            if (header.dwType != RIM_TYPEKEYBOARD || header.hDevice != m_selectedKeyboardHandle) {
                // 選択外のデバイス、またはキーボード以外は無視
                *result = 0;
                return false;
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
            // MakeCode + Flags を送る（文字変換はしない）
            EnqueueKeyboardRawEvent((uint16_t)k.MakeCode, (uint16_t)k.Flags);

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
    rid.dwFlags = RIDEV_DEVNOTIFY;                    // WM_INPUT_DEVICE_CHANGE を受ける
    rid.hwndTarget = hwnd;

    RegisterRawInputDevices(&rid, 1, sizeof(rid));
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
            QString instanceId;
            if (TryDeriveDeviceInstanceId(devicePath, instanceId)) {
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

    for (int i = 0; i < static_cast<int>(keyboards.size()); ++i) {
        const auto& k = keyboards[i];

        const QString vidStr = k.hasVidPid ? Hex4(k.vid) : QStringLiteral("N/A");
        const QString pidStr = k.hasVidPid ? Hex4(k.pid) : QStringLiteral("N/A");

        const QString label = QStringLiteral(u"キーボード%1 (VID:%2 PID:%3)")
            .arg(i + 1)
            .arg(vidStr)
            .arg(pidStr);

        // userData に devicePath を入れて選択維持に使う
        ui.comboBox->addItem(label, k.devicePath);
        ui.comboBox->setItemData(i, k.devicePath, Qt::ToolTipRole);
        // UserRole + 1 に HANDLE を隠し持つ
        ui.comboBox->setItemData(i, QVariant::fromValue((void*)k.hDevice), Qt::UserRole + 1);
    }

    // 選択維持
    if (!prevPath.isEmpty()) {
        for (int i = 0; i < ui.comboBox->count(); ++i) {
            if (ui.comboBox->itemData(i).toString() == prevPath) {
                ui.comboBox->setCurrentIndex(i);
                return;
            }
        }
    }
    if (ui.comboBox->count() > 0) ui.comboBox->setCurrentIndex(0);
}
