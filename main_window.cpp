#include "main_window.h"
#include <QCloseEvent>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <windows.h>
#include "AppShutdown.h"
#include "window.h"
#include "Globals.h"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    ui.setupUi(this);

    // 初期値サイズ 1504*846 (16:9)
    resize(1504, 846);
    setMinimumSize(1504, 846);

    // centralwidgetにレイアウトを追加して各ウィジェットを適切に配置
    if (ui.centralwidget) {
        QHBoxLayout* mainLayout = new QHBoxLayout(ui.centralwidget);
        mainLayout->setContentsMargins(10, 10, 10, 10);
        mainLayout->setSpacing(10);

        // 左側にタブウィジェットを追加
        if (ui.tabWidget) {
            mainLayout->addWidget(ui.tabWidget);
        }

        // 右側に操作パネル用のレイアウトを作成
        QVBoxLayout* sideLayout = new QVBoxLayout();
        sideLayout->setSpacing(10);

        if (ui.groupBox) {
            sideLayout->addWidget(ui.groupBox);
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
        // 左上を(16, 5)に配置。上下左右20pxずつのサイズ差（合計40px）を維持するためのマージン設定。
        tabLayout->setContentsMargins(16, 5, 24, 35);
        // マージンで位置を固定するため、AlignCenterを削除して追加
        tabLayout->addWidget(ui.frame, 0, 0);
    }
}

MainWindow::~MainWindow() {}

RenderHostWidgets* MainWindow::getRenderFrame() const {
    return ui.frame;
}

void MainWindow::closeEvent(QCloseEvent *event) {
    RequestShutdown();
    QMainWindow::closeEvent(event);
}

void MainWindow::resizeEvent(QResizeEvent *event) {
    // 基本クラスのイベントを呼び出してレイアウトを更新
    QMainWindow::resizeEvent(event);

    // リサイズ中に描写が止まらないよう、フレーム描画を強制的に1回呼び出す
    RenderFrame();
}

bool MainWindow::nativeEvent(const QByteArray &eventType, void *message, qintptr *result) {
    MSG* msg = static_cast<MSG*>(message);
    if (msg->message == WM_SIZING) {
        RECT* rect = reinterpret_cast<RECT*>(msg->lParam);
        int w = rect->right - rect->left;
        int h = rect->bottom - rect->top;

        // 1504x846 (16:9) のウィンドウで 1280x720 のレンダー領域を確保する場合のオーバーヘッド
        const int horizontalOverhead = 224;
        const int verticalOverhead = 126;

        int renderW = w - horizontalOverhead;
        int renderH = h - verticalOverhead;

        int snappedRenderW, snappedRenderH;
        SnapToKnownResolution(renderW, renderH, snappedRenderW, snappedRenderH);

        int targetWindowW = snappedRenderW + horizontalOverhead;
        int targetWindowH = snappedRenderH + verticalOverhead;

        // ドラッグされている方向に応じて矩形を調整
        switch (msg->wParam) {
        case WMSZ_BOTTOM:
        case WMSZ_BOTTOMRIGHT:
        case WMSZ_RIGHT:
            rect->right = rect->left + targetWindowW;
            rect->bottom = rect->top + targetWindowH;
            break;
        case WMSZ_TOP:
        case WMSZ_TOPLEFT:
        case WMSZ_LEFT:
            rect->left = rect->right - targetWindowW;
            rect->top = rect->bottom - targetWindowH;
            break;
        case WMSZ_TOPRIGHT:
            rect->right = rect->left + targetWindowW;
            rect->top = rect->bottom - targetWindowH;
            break;
        case WMSZ_BOTTOMLEFT:
            rect->left = rect->right - targetWindowW;
            rect->bottom = rect->top + targetWindowH;
            break;
        }

        *result = TRUE;
        return true;
    } else if (msg->message == WM_ENTERSIZEMOVE) {
        g_isSizing = true;
    } else if (msg->message == WM_EXITSIZEMOVE) {
        g_isSizing = false;
        // リサイズ終了時に描画を確実に更新
        RenderFrame();
    }
    return QMainWindow::nativeEvent(eventType, message, result);
}
