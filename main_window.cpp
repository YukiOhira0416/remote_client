#include "main_window.h"
#include <QCloseEvent>
#include "AppShutdown.h"
#include "window.h"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    ui.setupUi(this);

    // 手動でリサイズ制御を行うため、自動レイアウト設定を削除
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
    QMainWindow::resizeEvent(event);
    if (ui.centralwidget && ui.frame) {
        int cw = ui.centralwidget->width();
        int ch = ui.centralwidget->height();
        // 左上のマージン(20, 20)を確保しつつ、16:9を維持して最大化
        int availableW = std::max(0, cw - 40);
        int availableH = std::max(0, ch - 40);
        int targetW, targetH;
        SnapToKnownResolution(availableW, availableH, targetW, targetH);
        ui.frame->setGeometry(20, 20, targetW, targetH);
    }
}
