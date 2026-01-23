#ifndef RENDERHOSTWIDGETS_H
#define RENDERHOSTWIDGETS_H

#include <QFrame>
#include <QResizeEvent>
#include <QSizePolicy>
#include <windows.h>
#include "Globals.h"
#include "window.h"

class RenderHostWidgets : public QFrame {
    Q_OBJECT
public:
    explicit RenderHostWidgets(QWidget *parent = nullptr) : QFrame(parent) {
        // ネイティブウィンドウとしてマークし、子ウィンドウの親になれるようにする
        setAttribute(Qt::WA_NativeWindow);
        // Qtの描画システムを介さず直接描写する場合の設定
        setAttribute(Qt::WA_PaintOnScreen);
        setAttribute(Qt::WA_NoSystemBackground);

        // 枠線を消して中身を埋め尽くすように設定
        setFrameShape(QFrame::NoFrame);
        setContentsMargins(0, 0, 0, 0);

        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    }

    // 16:9のアスペクト比を維持するための設定
    bool hasHeightForWidth() const override { return true; }
    int heightForWidth(int w) const override { return w * 9 / 16; }

    QSize sizeHint() const override {
        return QSize(1280, 720);
    }

    HWND getHostHwnd() const {
        return (HWND)winId();
    }

    // 子HWND（描写ウインドウ）のサイズを同期し、サーバーへ通知する
    void syncChildWindow() {
        if (g_hWnd) {
            // GetClientRectを使って物理ピクセルでのサイズを取得（High DPI対策）
            RECT rc;
            if (GetClientRect((HWND)winId(), &rc)) {
                int physicalW = rc.right - rc.left;
                int physicalH = rc.bottom - rc.top;

                // 子HWND（描写ウインドウ）のサイズを同期
                MoveWindow(g_hWnd, 0, 0, physicalW, physicalH, TRUE);
                // サーバーへ解像度変更を通知
                NotifyResolutionChange(physicalW, physicalH);
            }
        }
    }

protected:
    void resizeEvent(QResizeEvent *event) override {
        QFrame::resizeEvent(event);
        syncChildWindow();
    }
};

#endif // RENDERHOSTWIDGETS_H
