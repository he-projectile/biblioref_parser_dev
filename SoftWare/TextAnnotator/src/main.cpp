#include "MainWindow.h"

#include <QApplication>

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    QApplication::setApplicationName(
        "Text Annotator"
    );

    QApplication::setOrganizationName(
        "TextAnnotator"
    );

    MainWindow window;

    window.show();

    return app.exec();
}