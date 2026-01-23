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
        // RenderHostWidgetsの左上を(16, 5)に配置し、タブを上下左右20px相当(合計40px)大きく保つマージン設定
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
    // RenderHostWidgetsのアスペクト比を16:9に維持するようにMainWindowをスナップさせる
    // 1504x846のときに1280x720のレンダー領域を確保する場合、固定のオーバーヘッドは横224px, 縦126pxとなる
    const int horizontalOverhead = 224;
    const int verticalOverhead = 126;

    int currentW = width();
    int currentH = height();

    int renderW = currentW - horizontalOverhead;
    int renderH = currentH - verticalOverhead;

    int snappedRenderW, snappedRenderH;
    SnapToKnownResolution(renderW, renderH, snappedRenderW, snappedRenderH);

    int targetWindowW = snappedRenderW + horizontalOverhead;
    int targetWindowH = snappedRenderH + verticalOverhead;

    if (currentW != targetWindowW || currentH != targetWindowH) {
        resize(targetWindowW, targetWindowH);
        return;
    }

    QMainWindow::resizeEvent(event);
    // 子ウィジェット（RenderHostWidgets等）のサイズ調整は、設定したレイアウトにより自動で行われます
}
