#pragma once

#include <QPlainTextEdit>
#include <QWheelEvent>

class AnnotationModel;

class AnnotationEditor : public QPlainTextEdit
{
    Q_OBJECT

public:
    explicit AnnotationEditor(QWidget* parent = nullptr);

    void setAnnotationModel(AnnotationModel* model);

    void refreshAnnotations();

    QString selectedText() const;

signals:
    void annotationClicked(int index);

protected:
    void mousePressEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;


private:
    AnnotationModel* m_model = nullptr;
};