#include "main_window.h"
#include <QCloseEvent>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include "AppShutdown.h"
#include "window.h"

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
            mainLayout->addWidget(ui.tabWidget);
        }

        // 右側に操作パネル用のレイアウトを作成
        QVBoxLayout* sideLayout = new QVBoxLayout();
        sideLayout->setSpacing(10);

        if (ui.groupBox) {
            ui.groupBox->setFixedWidth(159);
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
        tabLayout->setContentsMargins(20, 20, 20, 20); // 上下左右20pxの余白を設定
        // RenderHostWidgetsはheightForWidthを持つため、レイアウト内で16:9が維持されるように配置
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
        }
    }
    return QMainWindow::nativeEvent(eventType, message, result);
}
