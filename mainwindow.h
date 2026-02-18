/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 6.10.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTextEdit>
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
    QTextEdit *textEdit;
    QLabel *label;
    QPushButton *pushButton_2;
    QLabel *label_2;
    QLineEdit *lineEdit;
    QLabel *label_3;
    QLineEdit *lineEdit_2;
    QGroupBox *groupBox_3;
    QCheckBox *checkBox;
    QLabel *label_5;
    QLabel *label_4;
    QGroupBox *groupBox_4;
    QCheckBox *checkBox_2;
    QLabel *label_6;
    QGroupBox *groupBox_5;
    QCheckBox *checkBox_3;
    QLabel *label_7;
    QPushButton *pushButton_3;
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
        textEdit = new QTextEdit(tab_2);
        textEdit->setObjectName("textEdit");
        textEdit->setGeometry(QRect(30, 60, 341, 41));
        label = new QLabel(tab_2);
        label->setObjectName("label");
        label->setGeometry(QRect(30, 30, 151, 21));
        QFont font;
        font.setPointSize(12);
        label->setFont(font);
        pushButton_2 = new QPushButton(tab_2);
        pushButton_2->setObjectName("pushButton_2");
        pushButton_2->setGeometry(QRect(410, 60, 91, 41));
        pushButton_2->setFont(font);
        label_2 = new QLabel(tab_2);
        label_2->setObjectName("label_2");
        label_2->setGeometry(QRect(560, 20, 171, 31));
        label_2->setFont(font);
        lineEdit = new QLineEdit(tab_2);
        lineEdit->setObjectName("lineEdit");
        lineEdit->setGeometry(QRect(560, 60, 471, 171));
        sizePolicy.setHeightForWidth(lineEdit->sizePolicy().hasHeightForWidth());
        lineEdit->setSizePolicy(sizePolicy);
        label_3 = new QLabel(tab_2);
        label_3->setObjectName("label_3");
        label_3->setGeometry(QRect(40, 230, 221, 31));
        label_3->setFont(font);
        lineEdit_2 = new QLineEdit(tab_2);
        lineEdit_2->setObjectName("lineEdit_2");
        lineEdit_2->setGeometry(QRect(40, 260, 761, 131));
        groupBox_3 = new QGroupBox(tab_2);
        groupBox_3->setObjectName("groupBox_3");
        groupBox_3->setGeometry(QRect(40, 450, 631, 71));
        checkBox = new QCheckBox(groupBox_3);
        checkBox->setObjectName("checkBox");
        checkBox->setGeometry(QRect(40, 30, 78, 20));
        label_5 = new QLabel(groupBox_3);
        label_5->setObjectName("label_5");
        label_5->setGeometry(QRect(90, 20, 501, 31));
        label_4 = new QLabel(tab_2);
        label_4->setObjectName("label_4");
        label_4->setGeometry(QRect(40, 410, 221, 41));
        label_4->setFont(font);
        groupBox_4 = new QGroupBox(tab_2);
        groupBox_4->setObjectName("groupBox_4");
        groupBox_4->setGeometry(QRect(40, 550, 631, 71));
        checkBox_2 = new QCheckBox(groupBox_4);
        checkBox_2->setObjectName("checkBox_2");
        checkBox_2->setGeometry(QRect(40, 30, 78, 20));
        label_6 = new QLabel(groupBox_4);
        label_6->setObjectName("label_6");
        label_6->setGeometry(QRect(90, 20, 501, 31));
        groupBox_5 = new QGroupBox(tab_2);
        groupBox_5->setObjectName("groupBox_5");
        groupBox_5->setGeometry(QRect(40, 640, 631, 71));
        checkBox_3 = new QCheckBox(groupBox_5);
        checkBox_3->setObjectName("checkBox_3");
        checkBox_3->setGeometry(QRect(40, 30, 78, 20));
        label_7 = new QLabel(groupBox_5);
        label_7->setObjectName("label_7");
        label_7->setGeometry(QRect(90, 20, 501, 31));
        pushButton_3 = new QPushButton(tab_2);
        pushButton_3->setObjectName("pushButton_3");
        pushButton_3->setGeometry(QRect(710, 660, 91, 51));
        pushButton_3->setFont(font);
        tabWidget->addTab(tab_2, QString());
        groupBox = new QGroupBox(centralwidget);
        groupBox->setObjectName("groupBox");
        groupBox->setGeometry(QRect(1340, 30, 161, 161));
        radioButton = new QRadioButton(groupBox);
        radioButton->setObjectName("radioButton");
        radioButton->setGeometry(QRect(10, 36, 92, 20));
        radioButton->setChecked(true);
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

        tabWidget->setCurrentIndex(1);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "MainWindow", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab), QCoreApplication::translate("MainWindow", "Controll", nullptr));
        label->setText(QCoreApplication::translate("MainWindow", "Client Name", nullptr));
        pushButton_2->setText(QCoreApplication::translate("MainWindow", "Save", nullptr));
        label_2->setText(QCoreApplication::translate("MainWindow", "System Infomation", nullptr));
        label_3->setText(QCoreApplication::translate("MainWindow", "Announcement", nullptr));
        groupBox_3->setTitle(QCoreApplication::translate("MainWindow", "Low-speed", nullptr));
        checkBox->setText(QString());
        label_5->setText(QCoreApplication::translate("MainWindow", "Network speed: ~100 Mbps / Max resolution of client display area: 1920x1080", nullptr));
        label_4->setText(QCoreApplication::translate("MainWindow", "Mode Selection", nullptr));
        groupBox_4->setTitle(QCoreApplication::translate("MainWindow", "Medium-speed", nullptr));
        checkBox_2->setText(QString());
        label_6->setText(QCoreApplication::translate("MainWindow", "Network speed: 150-250 Mbps / Max resolution of client display area: 2560x1440", nullptr));
        groupBox_5->setTitle(QCoreApplication::translate("MainWindow", "Higt-speed", nullptr));
        checkBox_3->setText(QString());
        label_7->setText(QCoreApplication::translate("MainWindow", "Network speed: 300+ Mbps / Max resolution of client display area: 3840x2160", nullptr));
        pushButton_3->setText(QCoreApplication::translate("MainWindow", "Save", nullptr));
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
