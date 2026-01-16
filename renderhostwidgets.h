#ifndef RENDERHOSTWIDGETS_H
#define RENDERHOSTWIDGETS_H

#include <QFrame>
#include <QResizeEvent>
#include <windows.h>
#include "Globals.h"

class RenderHostWidgets : public QFrame {
    Q_OBJECT
public:
    explicit RenderHostWidgets(QWidget* parent = nullptr) : QFrame(parent) {
        // Ensure this widget has its own native window handle
        setAttribute(Qt::WA_NativeWindow);
    }

    HWND getHwnd() const {
        return (HWND)this->winId();
    }

protected:
    void resizeEvent(QResizeEvent* event) override {
        QFrame::resizeEvent(event);
        // Resize child HWND (g_hWnd) if it exists
        if (g_hWnd) {
            MoveWindow(g_hWnd, 0, 0, width(), height(), TRUE);
        }
    }
};

#endif // RENDERHOSTWIDGETS_H
