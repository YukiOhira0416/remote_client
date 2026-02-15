#include "DisplaySyncClient.h"
#include "Globals.h"     // DISPLAY_SYNC_SERVER_IP / DISPLAY_SYNC_SERVER_PORT
#include "DebugLog.h"

#include <QStringList>

DisplaySyncClient::DisplaySyncClient(QObject* parent)
    : QObject(parent)
    , m_socket(new QTcpSocket(this))
    , m_reconnectTimer(new QTimer(this))
    , m_displayCount(0)
    , m_activeIndex(-1)
{
    // Configure reconnect timer (5 seconds interval).
    m_reconnectTimer->setInterval(5000);
    m_reconnectTimer->setSingleShot(false);

    connect(m_socket, &QTcpSocket::connected,
            this, &DisplaySyncClient::onConnected);
    connect(m_socket, &QTcpSocket::disconnected,
            this, &DisplaySyncClient::onDisconnected);
    connect(m_socket, &QTcpSocket::readyRead,
            this, &DisplaySyncClient::onReadyRead);
    connect(m_socket, &QTcpSocket::errorOccurred,
            this, &DisplaySyncClient::onErrorOccurred);

    connect(m_reconnectTimer, &QTimer::timeout,
            this, &DisplaySyncClient::onReconnectTimeout);

    connectToServer();
}

void DisplaySyncClient::connectToServer()
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

    const QString host = QString::fromLatin1(DISPLAY_SYNC_SERVER_IP);
    const quint16 port = static_cast<quint16>(DISPLAY_SYNC_SERVER_PORT);

    DebugLog(L"DisplaySyncClient: connecting to " + host.toStdWString() +
             L":" + std::to_wstring(port));

    m_socket->connectToHost(host, port);
}

void DisplaySyncClient::setActiveDisplayFromUi(int index)
{
    if (!m_socket) {
        return;
    }

    if (index < 0 || index > 3) {
        return;
    }

    // Remember the last display index requested from the UI.
    m_activeIndex = index;

    if (m_socket->state() != QAbstractSocket::ConnectedState) {
        return;
    }

    QByteArray line("SELECT ");
    line.append(QByteArray::number(index));
    line.append('\n');

    qint64 written = m_socket->write(line);
    if (written != line.size()) {
        DebugLog(L"DisplaySyncClient: failed to write full SELECT command.");
    }
}


void DisplaySyncClient::onConnected()
{
    DebugLog(L"DisplaySyncClient: connected to server.");

    // Stop reconnect timer while we are connected.
    if (m_reconnectTimer->isActive()) {
        m_reconnectTimer->stop();
    }

    // If there is a previously selected display index (from UI or earlier state),
    // notify the server so that it can synchronize its active display.
    if (m_socket && m_socket->state() == QAbstractSocket::ConnectedState &&
        m_activeIndex >= 0 && m_activeIndex <= 3) {
        QByteArray line("SELECT ");
        line.append(QByteArray::number(m_activeIndex));
        line.append('\n');
        m_socket->write(line);
    }
}

void DisplaySyncClient::onDisconnected()

{
    DebugLog(L"DisplaySyncClient: disconnected from server.");

    // Clear internal state. The UI may choose how to react to these signals.
    if (m_displayCount != 0) {
        m_displayCount = 0;
        emit displayCountChanged(m_displayCount);
    }
    if (m_activeIndex != -1) {
        m_activeIndex = -1;
        emit activeDisplayChanged(m_activeIndex);
    }

    // Start reconnect attempts if not already running.
    if (!m_reconnectTimer->isActive()) {
        m_reconnectTimer->start();
    }
}

void DisplaySyncClient::onReadyRead()
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

        line = line.trimmed();
        if (!line.isEmpty()) {
            processLine(line);
        }
    }
}

void DisplaySyncClient::onErrorOccurred(QAbstractSocket::SocketError socketError)
{
    Q_UNUSED(socketError);
    // Log and let reconnect timer handle retries.
    DebugLog(L"DisplaySyncClient: socket error occurred.");

    if (m_socket && m_socket->state() == QAbstractSocket::UnconnectedState) {
        if (!m_reconnectTimer->isActive()) {
            m_reconnectTimer->start();
        }
    }
}

void DisplaySyncClient::onReconnectTimeout()
{
    DebugLog(L"DisplaySyncClient: reconnect timer fired, trying to reconnect.");
    connectToServer();
}

void DisplaySyncClient::processLine(const QByteArray& line)
{
    // Split by spaces.
    QList<QByteArray> parts = line.split(' ');
    if (parts.isEmpty()) {
        return;
    }

    QByteArray command = parts[0].toUpper();

    if (command == "DISPLAYS" && parts.size() >= 2) {
        bool ok = false;
        int count = parts[1].toInt(&ok);
        if (!ok) return;

        if (count < 0) count = 0;
        if (count > 4) count = 4;

        if (count != m_displayCount) {
            m_displayCount = count;
            emit displayCountChanged(m_displayCount);
        }
    }
    else if (command == "ACTIVE" && parts.size() >= 2) {
        bool ok = false;
        int index = parts[1].toInt(&ok);
        if (!ok) return;

        if (index < -1) index = -1;
        if (index > 3) index = 3;

        if (index != m_activeIndex) {
            m_activeIndex = index;
            emit activeDisplayChanged(m_activeIndex);
        }
    }
    else if (command == "STATE" && parts.size() >= 3) {
        bool okCount = false;
        bool okIndex = false;
        int count = parts[1].toInt(&okCount);
        int index = parts[2].toInt(&okIndex);

        if (okCount) {
            if (count < 0) count = 0;
            if (count > 4) count = 4;

            if (count != m_displayCount) {
                m_displayCount = count;
                emit displayCountChanged(m_displayCount);
            }
        }

        if (okIndex) {
            if (index < -1) index = -1;
            if (index > 3) index = 3;

            if (index != m_activeIndex) {
                m_activeIndex = index;
                emit activeDisplayChanged(m_activeIndex);
            }
        }
    }
    else {
        // Unknown command; ignore silently.
    }
}
