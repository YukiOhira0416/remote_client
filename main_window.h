#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <QMainWindow>
#include <QResizeEvent>
#include <array>
#include <QString>
#include "mainwindow.h"
#include "renderhostwidgets.h"

class DisplaySyncClient;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    virtual ~MainWindow();

    RenderHostWidgets* getRenderFrame() const;

protected:
    void closeEvent(QCloseEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;

private slots:
    void on_pushButton_clicked();
    void onActiveDisplayChanged(int index);
    void onDisplayCountChanged(int count);

private:
    void initializeDisplaySelectionUi();
    void updateDisplayLabels(int activeDisplayCount);
    void applyActiveDisplayToUi(int activeIndex);

    Ui::MainWindow ui;
    DisplaySyncClient* m_displaySyncClient = nullptr;
    std::array<QString, 4> m_displayBaseLabels{};
    bool m_updatingDisplayFromServer = false;
};

#endif // MAIN_WINDOW_H
