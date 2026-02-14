/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 6.9.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "renderhostwidgets.h"

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QTabWidget *tabWidget;
    QWidget *tab;
    RenderHostWidgets *frame;
    QWidget *tab_2;
    QGroupBox *groupBox;
    QRadioButton *radioButton;
    QRadioButton *radioButton_2;
    QRadioButton *radioButton_3;
    QRadioButton *radioButton_4;
    QGroupBox *groupBox_2;
    QWidget *verticalLayoutWidget;
    QVBoxLayout *verticalLayout;
    QRadioButton *radioButton_5;
    QRadioButton *radioButton_6;
    QRadioButton *radioButton_7;
    QRadioButton *radioButton_9;
    QPushButton *pushButton;
    QMenuBar *menubar;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName("MainWindow");
        MainWindow->resize(1504, 846);
        QSizePolicy sizePolicy(QSizePolicy::Policy::Expanding, QSizePolicy::Policy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(MainWindow->sizePolicy().hasHeightForWidth());
        MainWindow->setSizePolicy(sizePolicy);
        MainWindow->setMinimumSize(QSize(1502, 845));
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName("centralwidget");
        tabWidget = new QTabWidget(centralwidget);
        tabWidget->setObjectName("tabWidget");
        tabWidget->setGeometry(QRect(10, 10, 1320, 760));
        tab = new QWidget();
        tab->setObjectName("tab");
        frame = new RenderHostWidgets(tab);
        frame->setObjectName("frame");
        frame->setGeometry(QRect(16, 5, 1280, 720));
        sizePolicy.setHeightForWidth(frame->sizePolicy().hasHeightForWidth());
        frame->setSizePolicy(sizePolicy);
        frame->setFrameShape(QFrame::Shape::StyledPanel);
        frame->setFrameShadow(QFrame::Shadow::Raised);
        tabWidget->addTab(tab, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName("tab_2");
        tabWidget->addTab(tab_2, QString());
        groupBox = new QGroupBox(centralwidget);
        groupBox->setObjectName("groupBox");
        groupBox->setGeometry(QRect(1340, 30, 161, 161));
        radioButton = new QRadioButton(groupBox);
        radioButton->setObjectName("radioButton");
        radioButton->setGeometry(QRect(10, 36, 92, 20));
        radioButton_2 = new QRadioButton(groupBox);
        radioButton_2->setObjectName("radioButton_2");
        radioButton_2->setGeometry(QRect(10, 64, 92, 20));
        radioButton_3 = new QRadioButton(groupBox);
        radioButton_3->setObjectName("radioButton_3");
        radioButton_3->setGeometry(QRect(10, 92, 92, 20));
        radioButton_4 = new QRadioButton(groupBox);
        radioButton_4->setObjectName("radioButton_4");
        radioButton_4->setGeometry(QRect(10, 120, 92, 20));
        groupBox_2 = new QGroupBox(centralwidget);
        groupBox_2->setObjectName("groupBox_2");
        groupBox_2->setGeometry(QRect(1340, 210, 151, 231));
        verticalLayoutWidget = new QWidget(groupBox_2);
        verticalLayoutWidget->setObjectName("verticalLayoutWidget");
        verticalLayoutWidget->setGeometry(QRect(10, 20, 131, 201));
        verticalLayout = new QVBoxLayout(verticalLayoutWidget);
        verticalLayout->setObjectName("verticalLayout");
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        radioButton_5 = new QRadioButton(verticalLayoutWidget);
        radioButton_5->setObjectName("radioButton_5");
        radioButton_5->setChecked(true);

        verticalLayout->addWidget(radioButton_5);

        radioButton_6 = new QRadioButton(verticalLayoutWidget);
        radioButton_6->setObjectName("radioButton_6");

        verticalLayout->addWidget(radioButton_6);

        radioButton_7 = new QRadioButton(verticalLayoutWidget);
        radioButton_7->setObjectName("radioButton_7");

        verticalLayout->addWidget(radioButton_7);

        radioButton_9 = new QRadioButton(verticalLayoutWidget);
        radioButton_9->setObjectName("radioButton_9");

        verticalLayout->addWidget(radioButton_9);

        pushButton = new QPushButton(verticalLayoutWidget);
        pushButton->setObjectName("pushButton");

        verticalLayout->addWidget(pushButton);

        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName("menubar");
        menubar->setGeometry(QRect(0, 0, 1504, 33));
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName("statusbar");
        MainWindow->setStatusBar(statusbar);

        retranslateUi(MainWindow);

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "MainWindow", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab), QCoreApplication::translate("MainWindow", "Controll", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QCoreApplication::translate("MainWindow", "Settings", nullptr));
        groupBox->setTitle(QCoreApplication::translate("MainWindow", "Select Display", nullptr));
        radioButton->setText(QCoreApplication::translate("MainWindow", "Display 1", nullptr));
        radioButton_2->setText(QCoreApplication::translate("MainWindow", "Display 2", nullptr));
        radioButton_3->setText(QCoreApplication::translate("MainWindow", "Display 3", nullptr));
        radioButton_4->setText(QCoreApplication::translate("MainWindow", "Display 4", nullptr));
        groupBox_2->setTitle(QCoreApplication::translate("MainWindow", "Shortcut Key", nullptr));
        radioButton_5->setText(QCoreApplication::translate("MainWindow", "Win + L", nullptr));
        radioButton_6->setText(QCoreApplication::translate("MainWindow", "Win + G", nullptr));
        radioButton_7->setText(QCoreApplication::translate("MainWindow", "Win + Alt + B", nullptr));
        radioButton_9->setText(QCoreApplication::translate("MainWindow", "Ctrl + Alt + Delete", nullptr));
        pushButton->setText(QCoreApplication::translate("MainWindow", "Key Send", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // MAINWINDOW_H
