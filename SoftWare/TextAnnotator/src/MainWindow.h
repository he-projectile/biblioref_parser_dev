#pragma once

#include <QMainWindow>
#include <QVector>
#include <QString>

class QPushButton;
class QLabel;
class QListWidget;
class AnnotationEditor;
class AnnotationModel;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);

private slots:
    void openTextFile();
    void saveAnnotations();
    void loadAnnotations();

    void addAnnotation();

private:
    void createUi();

    void refreshAnnotationList();

    void setCurrentFile(
        const QString& filePath
    );

    QPushButton* createEntityButton(
        const QString& label
    );

private:
    AnnotationEditor* m_editor = nullptr;

    AnnotationModel* m_model = nullptr;

    QListWidget* m_annotationList = nullptr;

    QLabel* m_fileLabel = nullptr;

    QString m_currentTextFile;
    QString m_currentJsonFile;
};