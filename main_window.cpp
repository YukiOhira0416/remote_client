#include "main_window.h"
#include <QCloseEvent>
#include "AppShutdown.h"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    ui.setupUi(this);
}

MainWindow::~MainWindow() {}

RenderHostWidgets* MainWindow::getRenderFrame() const {
    return ui.frame;
}

void MainWindow::closeEvent(QCloseEvent *event) {
    RequestShutdown();
    QMainWindow::closeEvent(event);
}
