#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <QMainWindow>
#include <QResizeEvent>
#include <QtGlobal>
#include <QString>
#include <vector>
#include "mainwindow.h"
#include "renderhostwidgets.h"

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    virtual ~MainWindow();

    RenderHostWidgets* getRenderFrame() const;

protected:
    void closeEvent(QCloseEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
    bool nativeEvent(const QByteArray &eventType, void *message, qintptr *result) override;

private:
    Ui::MainWindow ui;

    struct KeyboardEntry {
        quint16 vid = 0xFFFF;
        quint16 pid = 0xFFFF;
        bool hasVidPid = false;
        QString devicePath; // RawInput device path (RIDI_DEVICENAME)
        QString uniqueKey;  // Deduplication key (ContainerId or normalized path)
    };

    void registerKeyboardDeviceNotifications();
    void scheduleKeyboardListRefresh();
    void refreshKeyboardList();
    std::vector<KeyboardEntry> enumerateKeyboards() const;

    bool m_keyboardRefreshPending = false;
};

#endif // MAIN_WINDOW_H
