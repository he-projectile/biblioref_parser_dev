#pragma once

#include <QMainWindow>
#include <QVector>
#include <QString>
#include <QStringList>

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

protected:
    void closeEvent(QCloseEvent* event) override;

private slots:
    void openTextFile();
    void saveAnnotations();
    void loadAnnotations();
    void handleEntityShortcut(int index);

    void addAnnotation(
        const QString& label
        );

private:
    QStringList m_entityTypes;

private:
    void createUi();

    void refreshAnnotationList();

    void setCurrentFile(
        const QString& filePath
        );

    // Сохраняет JSON. Если текущего JSON ещё нет,
    // показывает диалог выбора файла.
    bool saveAnnotationsToFile();

    // Проверяет наличие несохранённых изменений.
    // true  — можно продолжать действие.
    // false — действие нужно отменить.
    bool confirmSaveChanges();

    QPushButton* createEntityButton(
        const QString& label,
        const QString& shortcut
        );

private:
    AnnotationEditor* m_editor = nullptr;

    AnnotationModel* m_model = nullptr;

    QListWidget* m_annotationList = nullptr;

    QLabel* m_fileLabel = nullptr;

    QString m_currentTextFile;
    QString m_currentJsonFile;

    // Есть ли изменения после последнего сохранения JSON.
    bool m_dirty = false;
};