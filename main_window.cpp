#include "main_window.h"
#include <QCloseEvent>
#include <QVBoxLayout>
#include "AppShutdown.h"
#include "window.h"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    ui.setupUi(this);

    // 初期値サイズ 1502*845
    resize(1502, 845);

    // centralwidgetにレイアウトを追加してtabWidgetを追従させる
    if (ui.centralwidget) {
        QVBoxLayout* layout = new QVBoxLayout(ui.centralwidget);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(0);
        layout->addWidget(ui.tabWidget);
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
    // MainWindowのアスペクト比を16:9に維持する
    int w = width();
    int h = height();
    int tw, th;
    SnapToKnownResolution(w, h, tw, th);
    if (w != tw || h != th) {
        resize(tw, th);
        return;
    }

    QMainWindow::resizeEvent(event);

    // タブ内のRenderHostWidgetsのサイズと位置を調整
    if (ui.tabWidget && ui.frame) {
        QWidget* currentTab = ui.tabWidget->currentWidget();
        if (currentTab) {
            int tw = currentTab->width();
            int th = currentTab->height();

            int targetW, targetH;
            SnapToKnownResolution(tw, th, targetW, targetH);

            // 左上(0, 0)に合わせて配置
            ui.frame->setGeometry(0, 0, targetW, targetH);
        }
    }
}
