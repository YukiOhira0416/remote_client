#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <QMainWindow>
#include <QResizeEvent>
#include "mainwindow.h"
#include "renderhostwidgets.h"

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    virtual ~MainWindow();

    RenderHostWidgets* getRenderFrame() const;

protected:
    void closeEvent(QCloseEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
    bool nativeEvent(const QByteArray &eventType, void *message, qintptr *result) override;

private slots:
    void on_pushButton_clicked();

private:
    Ui::MainWindow ui;
};

#endif // MAIN_WINDOW_H
