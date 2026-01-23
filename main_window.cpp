#include "main_window.h"
#include <QCloseEvent>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include "AppShutdown.h"
#include "window.h"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    ui.setupUi(this);

    // 初期値サイズ 1504*846 (16:9)
    resize(1504, 846);
    setMinimumSize(1504, 846);

    // centralwidgetにレイアウトを追加して各ウィジェットを適切に配置
    if (ui.centralwidget) {
        QHBoxLayout* mainLayout = new QHBoxLayout(ui.centralwidget);
        mainLayout->setContentsMargins(10, 10, 10, 10);
        mainLayout->setSpacing(5);

        // 左側にタブウィジェットを追加
        if (ui.tabWidget) {
            mainLayout->addWidget(ui.tabWidget);
        }

        // 右側に操作パネル用のレイアウトを作成
        QVBoxLayout* sideLayout = new QVBoxLayout();
        sideLayout->setSpacing(10);

        if (ui.groupBox) {
            ui.groupBox->setFixedWidth(161);
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
        tabLayout->setContentsMargins(16, 20, 24, 20); // 垂直方向のマージンを均等化して中央に配置
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
    // 横: 余白(10+10) + タブ内余白(16+24=40) + 間隔(5) + サイドパネル(161) = 226
    // 縦: 余白(10+10) + タブ内余白(20+20=40) = 60
    const int oh_w = 226;
    const int oh_h = 60;

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
