#ifndef MODE_SYNC_CLIENT_H
#define MODE_SYNC_CLIENT_H

#include <QObject>
#include <QTcpSocket>
#include <QAbstractSocket>
#include <QTimer>
#include <QByteArray>

/// @brief Keeps the "Mode Selection" (Low/Medium/High speed) in sync
/// between this client and the task-tray application on the server.
///
/// Protocol (line-based, UTF-8, LF-terminated):
///   - Client -> Server:
///       MODE <n>\n      (n is 1, 2, or 3)
///   - Server -> Client:
///       MODE <n>\n      (n is 1, 2, or 3)
///
/// Any unknown or malformed lines are ignored.
class ModeSyncClient : public QObject
{
    Q_OBJECT
public:
    explicit ModeSyncClient(QObject* parent = nullptr);

    /// @brief Returns the last known mode (1=Low, 2=Medium, 3=High).
    int mode() const { return m_mode; }

    /// @brief Called from UI when the user presses Save after changing the mode.
    /// The value must be 1, 2, or 3. Out-of-range values are ignored.
    void setModeFromUi(int mode);

signals:
    /// @brief Emitted when the mode is updated from the server.
    void modeChanged(int mode);

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
    int         m_mode;
};

#endif // MODE_SYNC_CLIENT_H
