#include "ModeSyncClient.h"
#include "Globals.h"     // MODE_SYNC_SERVER_IP / MODE_SYNC_SERVER_PORT
#include "DebugLog.h"

#include <QStringList>
#include <QList>

ModeSyncClient::ModeSyncClient(QObject* parent)
    : QObject(parent)
    , m_socket(new QTcpSocket(this))
    , m_reconnectTimer(new QTimer(this))
    , m_mode(1)
{
    // Configure reconnect timer (5 seconds interval).
    m_reconnectTimer->setInterval(5000);
    m_reconnectTimer->setSingleShot(false);

    connect(m_socket, &QTcpSocket::connected,
            this, &ModeSyncClient::onConnected);
    connect(m_socket, &QTcpSocket::disconnected,
            this, &ModeSyncClient::onDisconnected);
    connect(m_socket, &QTcpSocket::readyRead,
            this, &ModeSyncClient::onReadyRead);
    connect(m_socket, &QTcpSocket::errorOccurred,
            this, &ModeSyncClient::onErrorOccurred);

    connect(m_reconnectTimer, &QTimer::timeout,
            this, &ModeSyncClient::onReconnectTimeout);

    connectToServer();
}

void ModeSyncClient::setModeFromUi(int mode)
{
    if (!m_socket) {
        return;
    }

    if (mode < 1 || mode > 3) {
        // Out-of-range; ignore.
        return;
    }

    // Remember the last mode requested from the UI.
    m_mode = mode;

    if (m_socket->state() != QAbstractSocket::ConnectedState) {
        // Connection is not ready yet; the server will push the latest
        // state again after reconnect, so we do not queue writes here.
        return;
    }

    QByteArray line("MODE ");
    line.append(QByteArray::number(mode));
    line.append('\n');

    qint64 written = m_socket->write(line);
    if (written != line.size()) {
        DebugLog(L"ModeSyncClient: failed to write full MODE command.");
    }
}

void ModeSyncClient::connectToServer()
{
    if (!m_socket) {
        return;
    }

    if (m_socket->state() == QAbstractSocket::ConnectedState ||
        m_socket->state() == QAbstractSocket::ConnectingState) {
        return;
    }

    // Abort any previous connection attempt and start a new one.
    m_socket->abort();

    const QString host = QString::fromLatin1(MODE_SYNC_SERVER_IP);
    const quint16 port = static_cast<quint16>(MODE_SYNC_SERVER_PORT);

    std::wstring msg = L"ModeSyncClient: connecting to server ";
    msg += host.toStdWString();
    msg += L":";
    msg += std::to_wstring(port);
    DebugLog(msg);

    m_socket->connectToHost(host, port);
}

void ModeSyncClient::onConnected()
{
    DebugLog(L"ModeSyncClient: connected to server.");

    // Stop reconnect timer while we are connected.
    if (m_reconnectTimer->isActive()) {
        m_reconnectTimer->stop();
    }

    // We do not send our current mode on connect; the server is the
    // source of truth and will push the current mode via a MODE line.
}

void ModeSyncClient::onDisconnected()
{
    DebugLog(L"ModeSyncClient: disconnected from server.");

    // Start reconnect attempts if not already running.
    if (!m_reconnectTimer->isActive()) {
        m_reconnectTimer->start();
    }
}

void ModeSyncClient::onReadyRead()
{
    if (!m_socket) {
        return;
    }

    m_receiveBuffer.append(m_socket->readAll());

    while (true) {
        int newlineIndex = m_receiveBuffer.indexOf('\n');
        if (newlineIndex < 0) {
            break; // Partial line; wait for more data.
        }

        QByteArray line = m_receiveBuffer.left(newlineIndex);
        m_receiveBuffer.remove(0, newlineIndex + 1);

        // Trim CR if present (handle CRLF).
        if (!line.isEmpty() && line.endsWith('\r')) {
            line.chop(1);
        }

        processLine(line);
    }
}

void ModeSyncClient::onErrorOccurred(QAbstractSocket::SocketError socketError)
{
    Q_UNUSED(socketError);
    // Log and let reconnect timer handle retries.
    DebugLog(L"ModeSyncClient: socket error occurred.");

    if (m_socket && m_socket->state() == QAbstractSocket::UnconnectedState) {
        if (!m_reconnectTimer->isActive()) {
            m_reconnectTimer->start();
        }
    }
}

void ModeSyncClient::onReconnectTimeout()
{
    DebugLog(L"ModeSyncClient: reconnect timer fired, trying to reconnect.");
    connectToServer();
}

void ModeSyncClient::processLine(const QByteArray& line)
{
    // Split by spaces.
    QList<QByteArray> parts = line.split(' ');
    if (parts.isEmpty()) {
        return;
    }

    QByteArray command = parts[0].toUpper();

    if (command == "MODE" && parts.size() >= 2) {
        bool ok = false;
        int mode = parts[1].toInt(&ok);
        if (!ok) {
            return;
        }

        if (mode < 1 || mode > 3) {
            return;
        }

        if (mode != m_mode) {
            m_mode = mode;
            emit modeChanged(m_mode);
        }
    }
    else {
        // Unknown command; ignore silently.
    }
}
