#ifndef DISPLAY_SYNC_CLIENT_H
#define DISPLAY_SYNC_CLIENT_H

#include <QObject>
#include <QTcpSocket>
#include <QAbstractSocket>
#include <QTimer>
#include <QByteArray>

// Forward declare DebugLog from DebugLog.h to avoid including it here.
void DebugLog(const std::wstring& message);

/// @brief Small helper that keeps the "Select Display" state in sync
/// between this client and the task-tray application on the server.
///
/// Protocol (line-based, UTF-8, LF-terminated):
///   - Server -> Client:
///       "DISPLAYS <count>\n"   : number of physical displays (0..4)
///       "ACTIVE <index>\n"     : 0-based active display index (or -1 for none)
///       "STATE <count> <index>\n" : combined update (optional)
///
///   - Client -> Server:
///       "SELECT <index>\n"     : user selected 0-based display index
///
/// The client clamps <count> to [0,4] and <index> to [-1,3].
class DisplaySyncClient : public QObject
{
    Q_OBJECT
public:
    explicit DisplaySyncClient(QObject* parent = nullptr);

    /// Called from UI when the user selects a different display.
    /// index is 0-based (0..3).
    void setActiveDisplayFromUi(int index);

signals:
    /// Emitted whenever the server notifies that the active display index changed.
    /// index is 0-based (0..3), or -1 if there is no active display.
    void activeDisplayChanged(int index);

    /// Emitted whenever the server notifies that the number of physical displays changed.
    /// count is clamped to [0,4].
    void displayCountChanged(int count);

private slots:
    void onConnected();
    void onDisconnected();
    void onReadyRead();
    void onErrorOccurred(QAbstractSocket::SocketError socketError);
    void onReconnectTimeout();

private:
    void connectToServer();
    void processLine(const QByteArray& line);

    QTcpSocket* m_socket;
    QTimer*     m_reconnectTimer;
    QByteArray  m_receiveBuffer;
    int         m_displayCount;
    int         m_activeIndex;
};

#endif // DISPLAY_SYNC_CLIENT_H
