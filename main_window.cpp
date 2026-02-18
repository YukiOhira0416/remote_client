#include "main_window.h"
#include <QRadioButton>
#include <QTimer>
#include "DisplaySyncClient.h"
#include <QCloseEvent>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QScrollArea>
#include <QScrollBar>
#include <QMargins>
#include <QTabBar>
#include <QFont>
#include <QSizePolicy>
#include "AppShutdown.h"
#include "window.h"
#include "Globals.h"
#include "RemoteKeyboard.h"

namespace {
    // 右側の操作パネル(Select Display / Shortcut Key)の基準幅
    // - 描写領域の右端とウインドウ右端の距離を少し広げつつ一定に保つ
    // - Display 3 (Disconnect) のラベルが切れないようにするための幅
    constexpr int kControlPanelWidth = 199;
}

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    ui.setupUi(this);

#ifdef _WIN32
    // RemoteKeyboard から WM_APP+1 を受け取るための HWND をグローバルに保持
    g_mainWindowHwnd = reinterpret_cast<HWND>(this->winId());
#endif

    // 初期値サイズ 1544*846 (16:9 領域 1280*720 を確保しつつ、
    // 右側の操作パネルに少し余裕を持たせる)
    // 横: 1280 + 264 = 1544, 縦: 720 + 126 = 846
    resize(1544, 846);
    setMinimumSize(1544, 400);

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

                // タブバーの見た目と操作性を向上させる:
                // - タブ全体の縦横サイズを拡大
                // - 「Controll」「Settings」タブの文字サイズを少し大きく
                // - タブの位置を少し下に下げる
                QFont tabFont = bar->font();
                if (tabFont.pointSize() > 0) {
                    tabFont.setPointSizeF(tabFont.pointSizeF() * 1.3);
                } else if (tabFont.pixelSize() > 0) {
                    tabFont.setPixelSize(static_cast<int>(tabFont.pixelSize() * 1.3));
                }
                bar->setFont(tabFont);

                // padding と margin を用いて、タブの縦横サイズを
                // デフォルト比でほぼ 2 倍に拡大しつつ、
                // タブバー全体を少し下にオフセットする。
                bar->setStyleSheet(
                    "QTabBar::tab {"
                    "  padding: 10px 30px;"   // 上下, 左右 -> クリック領域を拡大
                    "  margin-top: 6px;"      // タブを少し下にずらす
                    "}"
                );
            }
            mainLayout->addWidget(ui.tabWidget);
        }

        // 右側に操作パネル用のレイアウトを作成
        QVBoxLayout* sideLayout = new QVBoxLayout();
        sideLayout->setSpacing(10);

        if (ui.groupBox) {
            ui.groupBox->setFixedWidth(kControlPanelWidth);
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
        }

        if (ui.groupBox_2) {
            // Shortcut Key グループもレイアウト管理下に置く。
            // Designer では groupBox_2 直下に verticalLayoutWidget を絶対配置しているだけなので、
            // groupBox_2 にレイアウトを付けて横方向にきちんと広がるようにする。
            if (ui.groupBox_2->layout() == nullptr) {
                auto* gb2Layout = new QVBoxLayout(ui.groupBox_2);
                // タイトル領域の分だけ上マージンを多めに取る
                gb2Layout->setContentsMargins(10, 24, 10, 10);
                gb2Layout->setSpacing(6);

                if (ui.verticalLayoutWidget) {
                    ui.verticalLayoutWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
                    gb2Layout->addWidget(ui.verticalLayoutWidget);
                }
            }

            // Shortcut Key グループも Select Display と同じ幅で固定し、
            // 右側の操作パネル全体の幅を一定に保つ
            ui.groupBox_2->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Minimum);
            ui.groupBox_2->setFixedWidth(kControlPanelWidth);
        }

        // 上寄せで配置（間にstretchが入るため、意図した高さを維持したまま上に貼り付く）
        if (ui.groupBox) {
            sideLayout->addWidget(ui.groupBox, 0, Qt::AlignTop);
        }
        if (ui.groupBox_2) {
            sideLayout->addWidget(ui.groupBox_2, 0, Qt::AlignTop);
        }

        sideLayout->addStretch();

        mainLayout->addLayout(sideLayout);
        mainLayout->setStretch(0, 1); // tabWidgetを伸縮させる
        mainLayout->setStretch(1, 0); // 操作パネルは固定幅に近い扱い
    }

    // タブ内にスクロールエリアを配置し、その中にRenderHostWidgetsを入れる。
    // 描写領域は16:9を維持しつつ、横方向のリサイズに追従し、
    // 縦方向はスクロールで表示できるようにする。
    if (ui.tab && ui.frame) {
        m_renderScrollArea = new QScrollArea(ui.tab);
        m_renderScrollArea->setWidgetResizable(false);
        m_renderScrollArea->setFrameShape(QFrame::NoFrame);
        m_renderScrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        m_renderScrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        m_renderScrollArea->setWidget(ui.frame);

        QGridLayout* tabLayout = new QGridLayout(ui.tab);
        tabLayout->setContentsMargins(20, 20, 20, 20); // 上下左右20pxの余白を設定
        tabLayout->addWidget(m_renderScrollArea, 0, 0);
    }
    initializeDisplaySelectionUi();

    // 初期表示時に、ウインドウの横幅に合わせて描写領域サイズを16:9で調整
    updateRenderAreaByWidth();
}

MainWindow::~MainWindow() {}

RenderHostWidgets* MainWindow::getRenderFrame() const {
    return ui.frame;
}

void MainWindow::updateRenderAreaByWidth()
{
    if (!ui.frame)
        return;

    int availableWidth = 0;

    if (m_renderScrollArea && m_renderScrollArea->viewport()) {
        int viewportWidth = m_renderScrollArea->viewport()->width();

        // 垂直スクロールバーが表示されていないときにも、将来スクロールバーが出る
        // 分の右側の領域をあらかじめ確保しておく。
        //
        // - スクロールバー非表示時:
        //     viewportWidth = タブ領域の全幅
        //     描写領域幅 = viewportWidth - scrollExtent
        // - スクロールバー表示時:
        //     viewportWidth = タブ領域の全幅 - scrollExtent
        //     描写領域幅 = viewportWidth
        //
        // こうしておくことで、スクロールバーの有無に関わらず
        // 「描写領域の幅」は常に同じになり、スクロールバーが描写領域を
        // 覆わないようにできる。
        if (auto *vbar = m_renderScrollArea->verticalScrollBar()) {
            const int scrollExtent = vbar->sizeHint().width();
            if (!vbar->isVisible() && scrollExtent > 0) {
                viewportWidth -= scrollExtent;
            }
        }

        availableWidth = viewportWidth;
    } else if (ui.tab) {
        if (auto *layout = ui.tab->layout()) {
            const QMargins margins = layout->contentsMargins();
            availableWidth = ui.tab->width() - margins.left() - margins.right();
        } else {
            availableWidth = ui.tab->width();
        }
    } else if (auto *parent = ui.frame->parentWidget()) {
        availableWidth = parent->width();
    } else {
        availableWidth = ui.frame->width();
    }

    if (availableWidth <= 0)
        return;

    const int minRenderWidth  = 1280;
    // 最小幅1280の場合、16:9で自動的に高さ720になるため、minRenderHeightは不要

    int renderWidth = availableWidth;
    if (renderWidth < minRenderWidth) {
        renderWidth = minRenderWidth;
    }

    // 常に16:9のアスペクト比を維持（最小幅1280で高さは自動的に720以上になる）
    int renderHeight = renderWidth * 9 / 16;

    if (ui.frame->width() == renderWidth && ui.frame->height() == renderHeight)
        return;

    // 常に 16:9 の描写領域を維持できるように固定サイズを設定する。
    ui.frame->setFixedSize(renderWidth, renderHeight);
}

void MainWindow::closeEvent(QCloseEvent *event) {
    RequestShutdown();
    QMainWindow::closeEvent(event);
}

bool MainWindow::nativeEvent(const QByteArray &eventType, void *message, qintptr *result)
{
#ifdef _WIN32
    if (eventType == "windows_generic_MSG" || eventType == "windows_dispatcher_MSG") {
        MSG *msg = static_cast<MSG*>(message);
        if (msg && msg->message == WM_APP + 1) {
            // F1+F2 でフルスクリーン ⇔ 通常表示をトグル
            if (isFullScreen()) {
                showNormal();
            } else {
                showFullScreen();
            }

            if (result) {
                *result = 0;
            }
            return true;
        }
    }
#endif
    return QMainWindow::nativeEvent(eventType, message, result);
}

void MainWindow::resizeEvent(QResizeEvent *event) {
    QMainWindow::resizeEvent(event);

    // 利用可能な横幅（スクロールバーの有無を含む）が変化したときに
    // 描写領域がスクロールバーに隠れないよう、毎回サイズを更新する。
    //
    // ※ updateRenderAreaByWidth() は横幅だけを見て計算しているため、
    //    縦方向のリサイズでも viewport 幅が変わる（= スクロールバー出現）場合に
    //    正しく追従できる。
    updateRenderAreaByWidth();

    // RenderFrame を直接呼び出すと、内部での GPU 待ちやメッセージポンプ処理と
    // Qt のイベントディスパッチが再入してしまい、まれにウインドウ全体が固まる
    // ことがある。そのため、ここでは Qt のイベントループに 1 フレーム分だけ
    // 非同期で投げる。
    QTimer::singleShot(0, []() {
        RenderFrame();
    });
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
