#include "main_window.h"
#include <QCloseEvent>
#include <QVBoxLayout>
#include "AppShutdown.h"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    ui.setupUi(this);

    // RenderHostWidgetsがウィンドウサイズに合わせてリサイズされるようにレイアウトを設定
    if (!ui.centralwidget->layout()) {
        QVBoxLayout* layout = new QVBoxLayout(ui.centralwidget);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(0);
        layout->addWidget(ui.frame);
        // アスペクト比を維持するために中央配置
        layout->setAlignment(ui.frame, Qt::AlignCenter);
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
