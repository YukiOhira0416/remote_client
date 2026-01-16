#ifndef RENDERHOSTWIDGETS_H
#define RENDERHOSTWIDGETS_H

#include <QFrame>
#include <QResizeEvent>
#include <windows.h>
#include "Globals.h"

class RenderHostWidgets : public QFrame {
    Q_OBJECT
public:
    explicit RenderHostWidgets(QWidget *parent = nullptr) : QFrame(parent) {
        // ネイティブウィンドウとしてマークし、子ウィンドウの親になれるようにする
        setAttribute(Qt::WA_NativeWindow);
        // Qtの描画システムを介さず直接描写する場合の設定
        setAttribute(Qt::WA_PaintOnScreen);
        setAttribute(Qt::WA_NoSystemBackground);
    }

    HWND getHostHwnd() const {
        return (HWND)winId();
    }

protected:
    void resizeEvent(QResizeEvent *event) override {
        QFrame::resizeEvent(event);
        if (g_hWnd) {
            // 子HWND（描写ウインドウ）のサイズを同期
            MoveWindow(g_hWnd, 0, 0, width(), height(), TRUE);
        }
    }
};

#endif // RENDERHOSTWIDGETS_H
