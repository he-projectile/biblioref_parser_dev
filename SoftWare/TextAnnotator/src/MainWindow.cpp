#include "MainWindow.h"

#include "AnnotationEditor.h"
#include "AnnotationModel.h"
#include "JsonStorage.h"

#include <QAction>
#include <QFile>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QLabel>
#include <QListWidget>
#include <QMessageBox>
#include <QPushButton>
#include <QTextStream>
#include <QToolBar>
#include <QVBoxLayout>
#include <QWidget>
#include <QStatusBar>
#include <QFileInfo>
#include <QStatusBar>

#include <QTextCursor>
#include <QFileInfo>

#include <QCloseEvent>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    m_editor = new AnnotationEditor();


    createUi();

    connect(
        m_editor,
        &AnnotationEditor::entityShortcutPressed,
        this,
        &MainWindow::handleEntityShortcut
        );

    setWindowTitle("Text Annotator");

    resize(1200, 800);
}

void MainWindow::createUi()
{
    QWidget* centralWidget =
        new QWidget(this);

    setCentralWidget(centralWidget);

    QHBoxLayout* mainLayout =
        new QHBoxLayout(centralWidget);

    // ------------------------------------------------------------
    // Левая часть — текст
    // ------------------------------------------------------------

    QVBoxLayout* editorLayout =
        new QVBoxLayout();

    m_fileLabel =
        new QLabel("Файл не открыт");

    editorLayout->addWidget(m_fileLabel);

    m_editor->setAnnotationModel(m_model);

    editorLayout->addWidget(m_editor);

    mainLayout->addLayout(
        editorLayout,
        1
    );

    // ------------------------------------------------------------
    // Правая часть — управление
    // ------------------------------------------------------------

    QVBoxLayout* sideLayout =
        new QVBoxLayout();

    QLabel* entityLabel =
        new QLabel("Тип сущности:");

    sideLayout->addWidget(entityLabel);

    m_entityTypes =
        {
            "БИБЛ. ССЫЛКА",
            "АВТОРЫ",
            "НАЗВАНИЕ",
            "ГОД ПУБЛИКАЦИИ",
            "НАЗВАНИЕ ЖУРНАЛА",
            "НАЗВАНИЕ КОНФЕРЕНЦИИ",
            "СТРАНИЦЫ",
            "НОМЕР",
            "ИЗДАТЕЛЬСТВО",
            "МЕСТО ИЗДАНИЯ",

            "ТОМ",
            "DOI",
            "URL",
            "ISBN",
            "ISSN"
        };

    for (int i = 0; i < m_entityTypes.size(); ++i)
    {
        const QString& label =
            m_entityTypes[i];

        QString shortcut;

        if (i < 10)
        {
            shortcut =
                (i == 9)
                    ? "0"
                    : QString::number(i + 1);
        }

        QPushButton* button =
            createEntityButton(
                label,
                shortcut
                );

        sideLayout->addWidget(button);
    }


    sideLayout->addSpacing(20);

    QPushButton* addEntityButton =
        new QPushButton("+ Добавить тип");

    sideLayout->addWidget(addEntityButton);

    connect(
        addEntityButton,
        &QPushButton::clicked,
        this,
        [this, sideLayout]()
        {
            bool ok = false;

            const QString label =
                QInputDialog::getText(
                    this,
                    "Новый тип сущности",
                    "Название:",
                    QLineEdit::Normal,
                    "",
                    &ok
                    );

            if (!ok || label.trimmed().isEmpty())
                return;

            QPushButton* button =
                createEntityButton(
                    label.trimmed().toUpper()
                    , "");

            sideLayout->insertWidget(
                sideLayout->count() - 3,
                button
                );
        }
    );

    sideLayout->addSpacing(20);



    QLabel* annotationsLabel =
        new QLabel("Разметка:");

    sideLayout->addWidget(
        annotationsLabel
    );

    m_annotationList =
        new QListWidget();

    sideLayout->addWidget(
        m_annotationList,
        1
    );

    QPushButton* deleteButton =
        new QPushButton("Удалить разметку");

    sideLayout->addWidget(
        deleteButton
    );

    connect(
        deleteButton,
        &QPushButton::clicked,
        this,
        [this]()
        {
            const int row =
                m_annotationList->currentRow();

            if (row < 0)
                return;

            if (m_model->removeAnnotation(row))
            {
                m_dirty = true;
            }

            refreshAnnotationList();

            m_editor->refreshAnnotations();
        }
    );

    mainLayout->addLayout(
        sideLayout
    );

    // ------------------------------------------------------------
    // Верхнее меню
    // ------------------------------------------------------------

    QToolBar* toolbar =
        addToolBar("Файл");

    QAction* openAction =
        toolbar->addAction("Открыть TXT");

    QAction* saveAction =
        toolbar->addAction("Сохранить JSON");

    QAction* loadAction =
        toolbar->addAction("Загрузить JSON");

    connect(
        openAction,
        &QAction::triggered,
        this,
        &MainWindow::openTextFile
    );

    connect(
        saveAction,
        &QAction::triggered,
        this,
        &MainWindow::saveAnnotations
    );

    connect(
        loadAction,
        &QAction::triggered,
        this,
        &MainWindow::loadAnnotations
    );

    // ------------------------------------------------------------
    // Клик по существующей разметке
    // ------------------------------------------------------------

    connect(
        m_editor,
        &AnnotationEditor::annotationClicked,
        this,
        [this](int index)
        {
            m_annotationList->setCurrentRow(index);
        }
    );

    // ------------------------------------------------------------
    // Двойной клик по списку разметок
    // ------------------------------------------------------------

    connect(
        m_annotationList,
        &QListWidget::itemDoubleClicked,
        this,
        [this](QListWidgetItem* item)
        {
            Q_UNUSED(item);

            const int row =
                m_annotationList->currentRow();

            if (row < 0)
                return;

            const auto& annotations =
                m_model->annotations();

            if (row >= annotations.size())
                return;

            const Annotation& annotation =
                annotations[row];

            QTextCursor cursor =
                m_editor->textCursor();

            cursor.setPosition(
                annotation.start
            );

            cursor.setPosition(
                annotation.end,
                QTextCursor::KeepAnchor
            );

            m_editor->setTextCursor(cursor);

            m_editor->ensureCursorVisible();
        }
    );
}

QPushButton* MainWindow::createEntityButton(
    const QString& label,
    const QString& shortcut
    )
{
    QString buttonText = label;

    if (!shortcut.isEmpty())
    {
        buttonText =
            "[" + shortcut + "]  " + label;
    }

    QPushButton* button =
        new QPushButton(buttonText);

    button->setMinimumHeight(32);

    button->setProperty(
        "entityLabel",
        label
        );

    connect(
        button,
        &QPushButton::clicked,
        this,
        [this, label]()
        {
            addAnnotation(label);
        }
        );

    return button;
}

void MainWindow::openTextFile()
{
    if (!confirmSaveChanges())
        return;

    const QString filePath =
        QFileDialog::getOpenFileName(
            this,
            "Открыть TXT",
            QString(),
            "Text files (*.txt);;All files (*)"
        );

    if (filePath.isEmpty())
        return;

    QFile file(filePath);

    if (!file.open(QIODevice::ReadOnly |
                   QIODevice::Text))
    {
        QMessageBox::critical(
            this,
            "Ошибка",
            "Не удалось открыть файл."
        );

        return;
    }

    QTextStream stream(&file);

    const QString text =
        stream.readAll();

    file.close();

    m_editor->setPlainText(text);

    m_model->clear();

    m_editor->refreshAnnotations();

    refreshAnnotationList();

    setCurrentFile(filePath);
}

void MainWindow::saveAnnotations()
{
    saveAnnotationsToFile();
}

void MainWindow::loadAnnotations()
{
    if (!confirmSaveChanges())
        return;

    if (m_currentTextFile.isEmpty())
    {
        QMessageBox::warning(
            this,
            "Нет файла",
            "Сначала откройте TXT-файл."
        );

        return;
    }

    const QString filePath =
        QFileDialog::getOpenFileName(
            this,
            "Загрузить JSON",
            QString(),
            "JSON files (*.json)"
        );

    if (filePath.isEmpty())
        return;

    QString documentName;

    QVector<Annotation> annotations;

    QString error;

    if (!JsonStorage::load(
            filePath,
            documentName,
            annotations,
            &error))
    {
        QMessageBox::critical(
            this,
            "Ошибка",
            error
        );

        return;
    }

    const QString currentText =
        m_editor->toPlainText();

    for (const Annotation& annotation : annotations)
    {
        if (annotation.start < 0 ||
            annotation.end > currentText.length() ||
            annotation.start >= annotation.end)
        {
            QMessageBox::critical(
                this,
                "Ошибка",
                "JSON содержит диапазон, выходящий за пределы текста."
            );

            return;
        }

        const QString actualText =
            currentText.mid(
                annotation.start,
                annotation.end -
                annotation.start
            );

        if (actualText != annotation.text)
        {
            QMessageBox::critical(
                this,
                "Ошибка",
                QString(
                    "Текст annotation не соответствует "
                    "исходному тексту.\n\n"
                    "Ожидалось:\n%1\n\n"
                    "Получено:\n%2"
                )
                .arg(annotation.text)
                .arg(actualText)
            );

            return;
        }
    }

    m_model->clear();

    for (const Annotation& annotation : annotations)
    {
        if (!m_model->addAnnotation(annotation))
        {
            QMessageBox::critical(
                this,
                "Ошибка",
                "JSON содержит пересекающиеся разметки."
            );

            m_model->clear();

            return;
        }
    }

    m_editor->refreshAnnotations();

    refreshAnnotationList();

    m_currentJsonFile = filePath;

    m_dirty = false;

    statusBar()->showMessage(
        "Разметка загружена.",
        3000
    );
}

void MainWindow::addAnnotation(
    const QString& label
    )
{

    if (label.isEmpty())
        return;

    QTextCursor cursor =
        m_editor->textCursor();

    if (!cursor.hasSelection())
    {
        statusBar()->showMessage(
            "Сначала выделите текст.",
            2000
            );

        return;
    }

    const int start =
        cursor.selectionStart();

    const int end =
        cursor.selectionEnd();

    const QString text =
        cursor.selectedText();

    // ------------------------------------------------------------
    // REFERENCE
    // ------------------------------------------------------------

    if (label == "БИБЛ. ССЫЛКА")
    {
        // REFERENCE не может пересекаться
        // с существующей разметкой.
        if (m_model->hasAnyOverlap(start, end))
        {
            QMessageBox::warning(
                this,
                "Пересечение",
                "Библиографическая ссылка "
                "пересекается с существующей разметкой."
                );

            return;
        }
    }

    // ------------------------------------------------------------
    // Дочерняя сущность
    // ------------------------------------------------------------

    else
    {
        if (!m_model->isInsideReference(start, end))
        {
            QMessageBox::warning(
                this,
                "Нет библиографической ссылки",
                "Дочерние сущности нельзя разметить "
                "без выделенной библиографической ссылки.\n\n"
                "Сначала выделите всю библиографическую "
                "ссылку и назначьте ей "
                "\"БИБЛ. ССЫЛКА\"."
                );

            return;
        }

        // Дочерние сущности не должны пересекаться
        // друг с другом.
        if (m_model->hasChildOverlap(start, end))
        {
            QMessageBox::warning(
                this,
                "Пересечение",
                "Дочерняя сущность пересекается "
                "с другой дочерней сущностью."
                );

            return;
        }
    }

    // ------------------------------------------------------------
    // Создаём сущность
    // ------------------------------------------------------------

    Annotation annotation;

    annotation.start = start;
    annotation.end = end;
    annotation.label = label;
    annotation.text = text;

    if (!m_model->addAnnotation(annotation))
    {
        QMessageBox::warning(
            this,
            "Ошибка",
            "Не удалось добавить разметку."
            );

        return;
    }

    m_editor->refreshAnnotations();

    refreshAnnotationList();

    m_dirty = true;
}

void MainWindow::refreshAnnotationList()
{
    m_annotationList->clear();

    const auto& annotations =
        m_model->annotations();

    for (const Annotation& annotation : annotations)
    {
        const QString itemText =
            QString("[%1] %2")
                .arg(annotation.label)
                .arg(annotation.text);

        QListWidgetItem* item =
            new QListWidgetItem(itemText);

        item->setToolTip(
            QString(
                "start: %1\n"
                "end: %2\n"
                "label: %3\n"
                "text: %4"
            )
            .arg(annotation.start)
            .arg(annotation.end)
            .arg(annotation.label)
            .arg(annotation.text)
        );

        m_annotationList->addItem(item);
    }
}

void MainWindow::setCurrentFile(
    const QString& filePath
)
{
    m_currentTextFile = filePath;

    m_fileLabel->setText(
        "Файл: " +
        QFileInfo(filePath).fileName()
    );

    setWindowTitle(
        "Text Annotator — " +
        QFileInfo(filePath).fileName()
    );
}

bool MainWindow::saveAnnotationsToFile()
{
    if (m_currentTextFile.isEmpty())
    {
        QMessageBox::warning(
            this,
            "Нет файла",
            "Сначала откройте TXT-файл."
            );

        return false;
    }

    QString filePath =
        m_currentJsonFile;

    // Если JSON ещё не создавался,
    // спрашиваем путь.
    if (filePath.isEmpty())
    {
        const QString defaultPath =
            m_currentTextFile.left(
                m_currentTextFile.lastIndexOf('.')
                ) + ".json";

        filePath =
            QFileDialog::getSaveFileName(
                this,
                "Сохранить JSON",
                defaultPath,
                "JSON files (*.json)"
                );

        if (filePath.isEmpty())
            return false;
    }

    if (!JsonStorage::save(
            filePath,
            QFileInfo(m_currentTextFile).fileName(),
            m_model->annotations()))
    {
        QMessageBox::critical(
            this,
            "Ошибка",
            "Не удалось сохранить JSON."
            );

        return false;
    }

    m_currentJsonFile = filePath;

    m_dirty = false;

    statusBar()->showMessage(
        "Разметка сохранена.",
        3000
        );

    return true;
}

bool MainWindow::confirmSaveChanges()
{
    if (!m_dirty)
        return true;

    QMessageBox messageBox(
        QMessageBox::Warning,
        "Несохранённая разметка",
        "Разметка была изменена, но ещё не сохранена в JSON.\n\n"
        "Сохранить изменения?",
        QMessageBox::Save |
            QMessageBox::Discard |
            QMessageBox::Cancel,
        this
        );

    messageBox.setDefaultButton(
        QMessageBox::Save
        );

    const int result =
        messageBox.exec();

    if (result == QMessageBox::Save)
    {
        return saveAnnotationsToFile();
    }

    if (result == QMessageBox::Discard)
    {
        return true;
    }

    // Cancel
    return false;
}

void MainWindow::closeEvent(QCloseEvent* event)
{
    if (confirmSaveChanges())
    {
        event->accept();
    }
    else
    {
        event->ignore();
    }
}


void MainWindow::handleEntityShortcut(int index)
{
    if (index < 0 ||
        index >= m_entityTypes.size() ||
        index >= 10)
    {
        return;
    }

    addAnnotation(
        m_entityTypes[index]
        );
}
