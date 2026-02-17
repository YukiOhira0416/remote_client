#include "main_window.h"
#include <QRadioButton>
#include "DisplaySyncClient.h"
#include <QCloseEvent>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QTabBar>
#include <QSizePolicy>
#include "AppShutdown.h"
#include "window.h"
#include "RemoteKeyboard.h"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    ui.setupUi(this);

    // 初期値サイズ 1504*846 (16:9 領域 1280*720 を確保)
    // 横: 1280 + 224 = 1504, 縦: 720 + 126 = 846
    resize(1504, 846);
    setMinimumSize(1504, 846);

    // centralwidgetにレイアウトを追加して各ウィジェットを適切に配置
    if (ui.centralwidget) {
        QHBoxLayout* mainLayout = new QHBoxLayout(ui.centralwidget);
        mainLayout->setContentsMargins(10, 10, 10, 10);
        mainLayout->setSpacing(5);

        // 左側にタブウィジェットを追加
        if (ui.tabWidget) {
            ui.tabWidget->setDocumentMode(true);
            // タブバーとページ領域の境界線（QTabBarのベースライン）を描画しない
            // QTabWidget::tabBar() は protected のため findChild で取得する
            if (auto *bar = ui.tabWidget->findChild<QTabBar*>()) {
                bar->setDrawBase(false);
            }
            mainLayout->addWidget(ui.tabWidget);
        }

        // 右側に操作パネル用のレイアウトを作成
        QVBoxLayout* sideLayout = new QVBoxLayout();
        sideLayout->setSpacing(10);

        if (ui.groupBox) {
            ui.groupBox->setFixedWidth(159);
            // 重要: groupBox配下が"絶対配置"(geometry)のみだと、外側レイアウトに組み込んだ瞬間
            //       groupBoxのsizeHintが極小になり、タイトル行だけの高さに潰れて中のRadioButtonが
            //       クリップされることがある。
            //       → groupBox内部にもレイアウトを付け、sizeHint/最小高さがRadioButtonを含むようにする。
            if (ui.groupBox->layout() == nullptr) {
                auto* gbLayout = new QVBoxLayout(ui.groupBox);
                // タイトル領域の分だけ上マージンを多めに取る
                gbLayout->setContentsMargins(10, 24, 10, 10);
                gbLayout->setSpacing(6);

                auto addIf = [&](QWidget* w) {
                    if (w) gbLayout->addWidget(w);
                };
                addIf(ui.radioButton);
                addIf(ui.radioButton_2);
                addIf(ui.radioButton_3);
                addIf(ui.radioButton_4);

                // groupBox自体が縮み過ぎないよう、RadioButtonのsizeHintから最小高さを計算して設定
                int visibleCount = 0;
                int minH = gbLayout->contentsMargins().top() + gbLayout->contentsMargins().bottom();
                for (auto* rb : {ui.radioButton, ui.radioButton_2, ui.radioButton_3, ui.radioButton_4}) {
                    if (!rb) continue;
                    ++visibleCount;
                    minH += rb->sizeHint().height();
                }
                if (visibleCount >= 2) {
                    minH += gbLayout->spacing() * (visibleCount - 1);
                }
                // フレーム/スタイルの誤差吸収
                minH += 8;

                // Designer上の想定サイズ(高さ161)を下回らないようにする
                if (minH < 161) {
                    minH = 161;
                }

                ui.groupBox->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Minimum);
                ui.groupBox->setMinimumHeight(minH);
            }

            // 上寄せで配置（間にstretchが入るため、意図した高さを維持したまま上に貼り付く）
            sideLayout->addWidget(ui.groupBox, 0, Qt::AlignTop);
        }

        sideLayout->addStretch();

        mainLayout->addLayout(sideLayout);
        mainLayout->setStretch(0, 1); // tabWidgetを伸縮させる
        mainLayout->setStretch(1, 0); // 操作パネルは固定幅に近い扱い
    }

    // タブ内にレイアウトを設定してRenderHostWidgetsをアスペクト比維持で配置
    if (ui.tab && ui.frame) {
        QGridLayout* tabLayout = new QGridLayout(ui.tab);
        tabLayout->setContentsMargins(20, 20, 20, 20); // 上下左右20pxの余白を設定
        // RenderHostWidgetsはheightForWidthを持つため、レイアウト内で16:9が維持されるように配置
        tabLayout->addWidget(ui.frame, 0, 0);
    }
    initializeDisplaySelectionUi();
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

    // リサイズ中も描画を継続するためにRenderFrameを呼び出す
    // window.cpp側で非ブロッキング化されているため安全に呼び出せる
    RenderFrame();
}


void MainWindow::on_pushButton_clicked()
{
    // Shortcut Key group:
    //   radioButton_5: Win + L
    //   radioButton_6: Win + G
    //   radioButton_7: Win + Alt + B
    //   radioButton_9: Ctrl + Alt + Delete
    //
    // If no radio button is selected, do nothing.
    if (ui.radioButton_5 && ui.radioButton_5->isChecked()) {
        SendShortcutWinL();
    } else if (ui.radioButton_6 && ui.radioButton_6->isChecked()) {
        SendShortcutWinG();
    } else if (ui.radioButton_7 && ui.radioButton_7->isChecked()) {
        SendShortcutWinAltB();
    } else if (ui.radioButton_9 && ui.radioButton_9->isChecked()) {
        SendShortcutCtrlAltDel();
    }
}

void MainWindow::initializeDisplaySelectionUi()
{
    // Cache base labels for each display radio button (Display 1..4).
    QRadioButton* displayButtons[4] = {
        ui.radioButton,
        ui.radioButton_2,
        ui.radioButton_3,
        ui.radioButton_4
    };

    for (int i = 0; i < 4; ++i) {
        if (displayButtons[i]) {
            m_displayBaseLabels[i] = displayButtons[i]->text();
        } else {
            m_displayBaseLabels[i].clear();
        }
    }

    // Connect radio buttons -> DisplaySyncClient
    auto connectDisplayRadio = [this](QRadioButton* button, int index) {
        if (!button) {
            return;
        }
        connect(button, &QRadioButton::toggled, this,
                [this, index](bool checked) {
                    if (!checked) {
                        return;
                    }
                    if (m_updatingDisplayFromServer) {
                        // Avoid feedback loops when we are applying state from the server.
                        return;
                    }
                    if (m_displaySyncClient) {
                        m_displaySyncClient->setActiveDisplayFromUi(index);
                    }
                });
    };

    connectDisplayRadio(ui.radioButton,   0);
    connectDisplayRadio(ui.radioButton_2, 1);
    connectDisplayRadio(ui.radioButton_3, 2);
    connectDisplayRadio(ui.radioButton_4, 3);

    // Create DisplaySyncClient and wire its signals to this window.
    m_displaySyncClient = new DisplaySyncClient(this);

    connect(m_displaySyncClient, &DisplaySyncClient::activeDisplayChanged,
            this, &MainWindow::onActiveDisplayChanged);
    connect(m_displaySyncClient, &DisplaySyncClient::displayCountChanged,
            this, &MainWindow::onDisplayCountChanged);
}

void MainWindow::updateDisplayLabels(int activeDisplayCount)
{
    int count = activeDisplayCount;
    if (count < 0) {
        count = 0;
    }
    if (count > 4) {
        count = 4;
    }

    QRadioButton* displayButtons[4] = {
        ui.radioButton,
        ui.radioButton_2,
        ui.radioButton_3,
        ui.radioButton_4
    };

    for (int i = 0; i < 4; ++i) {
        QRadioButton* button = displayButtons[i];
        if (!button) {
            continue;
        }

        QString baseLabel;
        if (i < static_cast<int>(m_displayBaseLabels.size()) &&
            !m_displayBaseLabels[i].isEmpty()) {
            baseLabel = m_displayBaseLabels[i];
        } else {
            baseLabel = QStringLiteral("Display %1").arg(i + 1);
            m_displayBaseLabels[i] = baseLabel;
        }

        if (count > 0 && i >= count) {
            // There are fewer physical displays than the number of radio buttons.
            // For example: when server has 2 monitors, Display 3/4 will show "(Disconnect)".
            button->setText(baseLabel + QStringLiteral(" (Disconnect)"));
        } else {
            button->setText(baseLabel);
        }
    }
}

void MainWindow::applyActiveDisplayToUi(int activeIndex)
{
    m_updatingDisplayFromServer = true;

    QRadioButton* displayButtons[4] = {
        ui.radioButton,
        ui.radioButton_2,
        ui.radioButton_3,
        ui.radioButton_4
    };

    for (int i = 0; i < 4; ++i) {
        QRadioButton* button = displayButtons[i];
        if (!button) {
            continue;
        }
        bool shouldCheck = (i == activeIndex);
        if (button->isChecked() != shouldCheck) {
            button->setChecked(shouldCheck);
        }
    }

    m_updatingDisplayFromServer = false;
}

void MainWindow::onActiveDisplayChanged(int index)
{
    // index is 0-based; -1 means "no active display".
    applyActiveDisplayToUi(index);
}

void MainWindow::onDisplayCountChanged(int count)
{
    updateDisplayLabels(count);
}
