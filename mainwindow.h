#ifndef MAINWINDOW_ACTUAL_H
#define MAINWINDOW_ACTUAL_H

#include <QMainWindow>
#include "ui_mainwindow.h"
#include "renderhostwidgets.h"

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr) : QMainWindow(parent), ui(new Ui::MainWindow) {
        ui->setupUi(this);
    }
    ~MainWindow() {
        delete ui;
    }

    RenderHostWidgets* getRenderHost() const {
        return ui->frame;
    }

private:
    Ui::MainWindow* ui;
};

#endif // MAINWINDOW_ACTUAL_H
