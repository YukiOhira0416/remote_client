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

    // レイアウトを介さず絶対座標で配置するため、既存のレイアウト設定をクリア
    // ui.tabWidget にスタイルを適用して境界線の黒い線（pane border）を解消
    if (ui.tabWidget) {
        ui.tabWidget->setDocumentMode(true);
        ui.tabWidget->setStyleSheet("QTabWidget::pane { border: 0; }");
    }

    if (ui.groupBox) {
        ui.groupBox->setFixedWidth(159);
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

    // MainWindow A のサイズから RenderHostWidgets C のサイズを決定
    // 横オーバーヘッド: 224 (余白10*2 + サイドパネル159 + 間隔5 + タブ内余白等)
    // 縦オーバーヘッド: 126 (余白10*2 + メニュー・タブバー等)
    const int oh_w = 224;
    const int oh_h = 126;

    int targetW = width() - oh_w;
    int targetH = height() - oh_h;

    // C のサイズを 16:9 にスナップ（A が 16:9 を維持していれば自然に 16:9 になる）
    int cw, ch;
    SnapToKnownResolution(targetW, targetH, cw, ch);

    // MainWindow A 自体のサイズをスナップした C に合わせて調整 (アスペクト比維持の徹底)
    int nextW = cw + oh_w;
    int nextH = ch + oh_h;
    if (width() != nextW || height() != nextH) {
        resize(nextW, nextH);
        inResize = false;
        return;
    }

    // B (tabWidget) の配置とサイズ設定
    // B は A に対して (10, 10) に配置、サイズは C より上下左右 20px ずつ大きくする (+40)
    if (ui.tabWidget) {
        ui.tabWidget->move(10, 10);
        ui.tabWidget->resize(cw + 40, ch + 40);
    }

    // C (frame) の配置とサイズ設定
    // C は B (のページ) に対して (16, 5) に配置
    if (ui.frame) {
        ui.frame->move(16, 5);
        ui.frame->resize(cw, ch);
    }

    // サイドパネルのウィジェットを配置 (B の右側に 5px の間隔)
    int sideX = 10 + (cw + 40) + 5;
    if (ui.groupBox) {
        ui.groupBox->move(sideX, 10);
    }
    if (ui.label) {
        ui.label->move(sideX, height() - 86);
    }
    if (ui.comboBox) {
        ui.comboBox->move(sideX, height() - 50);
        ui.comboBox->setFixedWidth(159);
    }

    QMainWindow::resizeEvent(event);
    inResize = false;

    // リサイズ中も描画を継続
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
