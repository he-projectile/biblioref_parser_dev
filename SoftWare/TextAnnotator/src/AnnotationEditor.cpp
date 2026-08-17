#include "AnnotationEditor.h"

#include "AnnotationModel.h"

#include <QMouseEvent>
#include <QTextCursor>
#include <QTextCharFormat>
#include <QColor>
#include <QWheelEvent>
#include <QScrollBar>

static QColor colorForLabel(
    const QString& label
    )
{
    static const QVector<QColor> colors =
        {
            QColor("#FFE3E3"),
            QColor("#E3F0FF"), QColor("#E5F9E9"),
            QColor("#FFF6D6"), QColor("#F0E8FA"),
            QColor("#FFE7D1"), QColor("#E3FAFA"),
            QColor("#ECF6DC"), QColor("#FCE4E6"),
            QColor("#E8F2FA"), QColor("#F2ECFA"),
            QColor("#EAF9E3"), QColor("#FFF0FA"),
            QColor("#F9E2FA"), QColor("#E3FAF2")
        };

    uint hash = 0;

    for (const QChar& character : label)
    {
        hash = hash * 31 + character.unicode();
    }

    return colors[
        hash % colors.size()
    ];
}

AnnotationEditor::AnnotationEditor(QWidget* parent)
    : QPlainTextEdit(parent)
{
    setReadOnly(true);

    setLineWrapMode(QPlainTextEdit::NoWrap);

    QFont font("Consolas");
    font.setPointSize(11);

    setFont(font);
}

void AnnotationEditor::setAnnotationModel(AnnotationModel* model)
{
    m_model = model;

    refreshAnnotations();
}

QString AnnotationEditor::selectedText() const
{
    return textCursor().selectedText();
}

void AnnotationEditor::refreshAnnotations()
{
    QList<QTextEdit::ExtraSelection> selections;

    if (!m_model)
    {
        setExtraSelections(selections);
        return;
    }

    const QVector<Annotation>& annotations =
        m_model->annotations();

    // ============================================================
    // Слой 1. Обычные сущности
    // ============================================================

    for (const Annotation& annotation : annotations)
    {
        if (annotation.label == "БИБЛ. ССЫЛКА")
            continue;

        QTextCursor cursor(document());

        cursor.setPosition(annotation.start);

        cursor.setPosition(
            annotation.end,
            QTextCursor::KeepAnchor
            );

        QTextCharFormat format;

        format.setBackground(
            colorForLabel(annotation.label)
            );

        format.setForeground(
            Qt::black
            );

        QTextEdit::ExtraSelection selection;

        selection.cursor = cursor;
        selection.format = format;

        selections.append(selection);
    }

    // ============================================================
    // Слой 2. REFERENCE
    //
    // Добавляем ПОСЛЕ дочерних сущностей.
    // Поэтому underline должен отображаться поверх них.
    // ============================================================

    for (const Annotation& annotation : annotations)
    {
        if (annotation.label != "БИБЛ. ССЫЛКА")
            continue;

        QTextCursor cursor(document());

        cursor.setPosition(annotation.start);

        cursor.setPosition(
            annotation.end,
            QTextCursor::KeepAnchor
            );

        QTextCharFormat format;

        format.setUnderlineStyle(
            QTextCharFormat::SingleUnderline
            );

        format.setUnderlineColor(
            QColor("#555555")
            );

        QTextEdit::ExtraSelection selection;

        selection.cursor = cursor;
        selection.format = format;

        selections.append(selection);
    }

    setExtraSelections(selections);
}

void AnnotationEditor::mousePressEvent(QMouseEvent* event)
{
    QPlainTextEdit::mousePressEvent(event);

    if (!m_model)
        return;

    const QTextCursor cursor = textCursor();

    if (!cursor.hasSelection())
        return;

    const int position =
        cursor.selectionStart();

    const int index =
        m_model->annotationAt(position);

    if (index >= 0)
        emit annotationClicked(index);
}

void AnnotationEditor::wheelEvent(QWheelEvent* event)
{
    if (event->modifiers() & Qt::ShiftModifier)
    {
        QScrollBar* scrollBar =
            horizontalScrollBar();

        const int delta =
            event->angleDelta().y();

        scrollBar->setValue(
            scrollBar->value() - delta
            );

        event->accept();

        return;
    }

    QPlainTextEdit::wheelEvent(event);
}