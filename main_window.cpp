#include "main_window.h"
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
    // 再帰呼び出しを防ぐためのガード
    static bool inResize = false;
    if (inResize) return;
    inResize = true;

    // 内部レイアウトの固定オーバーヘッドを差し引いてレンダリング領域を計算
    // 横: 余白(10+10) + タブ内余白(20+20=40) + 間隔(5) + サイドパネル(159) = 224
    // 縦: 余白(10+10) + タブ内余白(20+20=40) + メニューバー・タブバー等(~66) = 126
    const int oh_w = 224;
    const int oh_h = 126;

    int targetW = width() - oh_w;
    int targetH = height() - oh_h;

    int snappedW, snappedH;
    SnapToKnownResolution(targetW, targetH, snappedW, snappedH);

    int nextW = snappedW + oh_w;
    int nextH = snappedH + oh_h;

    if (width() != nextW || height() != nextH) {
        resize(nextW, nextH);
        inResize = false;
        return;
    }

    QMainWindow::resizeEvent(event);
    inResize = false;

    // リサイズ中も描画を継続するためにRenderFrameを呼び出す
    // window.cpp側で非ブロッキング化されているため安全に呼び出せる
    RenderFrame();
}

bool MainWindow::nativeEvent(const QByteArray &eventType, void *message, qintptr *result) {
    if (eventType == "windows_generic_MSG") {
        MSG *msg = static_cast<MSG *>(message);
        if (msg->message == WM_NCHITTEST) {
            // デフォルトの挙動でどこがヒットしたか取得
            *result = DefWindowProc(msg->hwnd, msg->message, msg->wParam, msg->lParam);
            // 右下のリサイズハンドル(HTBOTTOMRIGHT)以外のリサイズハンドルを制限
            // 他の辺や角のリサイズハンドルが検出された場合、単なる境界(HTBORDER)として扱いリサイズを無効化する
            if (*result == HTLEFT || *result == HTRIGHT || *result == HTTOP ||
                *result == HTTOPLEFT || *result == HTTOPRIGHT ||
                *result == HTBOTTOM || *result == HTBOTTOMLEFT) {
                *result = HTBORDER;
                return true;
            }
        } else if (msg->message == WM_SIZING) {
            // 右下のリサイズのみを許可している前提で、アスペクト比16:9を維持
            RECT *rect = reinterpret_cast<RECT *>(msg->lParam);
            int w = rect->right - rect->left;

            // 幅に基づいて高さを16:9に調整
            int targetH = w * 9 / 16;

            // 最小サイズ(1504x846)を維持
            if (w < 1504) {
                w = 1504;
                targetH = 846;
                rect->right = rect->left + w;
            }

            rect->bottom = rect->top + targetH;

            *result = TRUE;
            return true;
        }
    }
    return QMainWindow::nativeEvent(eventType, message, result);
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
