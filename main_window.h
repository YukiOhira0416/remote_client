#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <QMainWindow>
#include <QResizeEvent>
#include <QtGlobal>
#include <QString>
#include <QMap>
#include <vector>
#include <windows.h>
#include "mainwindow.h"
#include "renderhostwidgets.h"

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    enum class BusType {
        BuiltIn = 0,
        USB = 1,
        Bluetooth = 2,
        Other = 3
    };

    explicit MainWindow(QWidget *parent = nullptr);
    virtual ~MainWindow();

    RenderHostWidgets* getRenderFrame() const;

protected:
    void closeEvent(QCloseEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
    bool nativeEvent(const QByteArray &eventType, void *message, qintptr *result) override;
    bool event(QEvent* event) override;

private:
    Ui::MainWindow ui;

    struct KeyboardEntry {
        HANDLE hDevice = nullptr;
        quint16 vid = 0xFFFF;
        quint16 pid = 0xFFFF;
        bool hasVidPid = false;
        QString devicePath; // RawInput device path (RIDI_DEVICENAME)
        QString uniqueKey;  // Deduplication key (ContainerId or normalized path)
        BusType busType = BusType::Other;
    };

    void registerKeyboardDeviceNotifications();
    void scheduleKeyboardListRefresh();
    void refreshKeyboardList();
    std::vector<KeyboardEntry> enumerateKeyboards() const;

    bool m_keyboardRefreshPending = false;
    QString m_selectedKeyboardPath; // combo selection cache
    QString m_selectedKeyboardUniqueKey;
    QMap<HANDLE, QString> m_handleToUniqueKey;
    HHOOK m_kbdHook = nullptr;
    HWND m_mainHwnd = nullptr;

    bool isClientFocusedWin32() const;
};

#endif // MAIN_WINDOW_H
