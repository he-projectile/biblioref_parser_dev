#pragma once

#include "Annotation.h"

#include <QVector>
#include <QString>

class AnnotationModel
{
public:
    void clear();

    const QVector<Annotation>& annotations() const;

    bool addAnnotation(
        const Annotation& annotation
        );

    bool removeAnnotation(
        int index
        );

    bool updateLabel(
        int index,
        const QString& label
        );

    int annotationAt(
        int position
        ) const;

    bool hasOverlap(
        int start,
        int end,
        int ignoreIndex = -1
        ) const;

    bool hasAnyOverlap(
        int start,
        int end
        ) const;

    bool hasChildOverlap(
        int start,
        int end
        ) const;

    bool isInsideReference(
        int start,
        int end
        ) const;

    int findParent(
        int start,
        int end
        ) const;

private:
    bool rangesOverlap(
        int start1,
        int end1,
        int start2,
        int end2
        ) const;

    bool rangeContains(
        int outerStart,
        int outerEnd,
        int innerStart,
        int innerEnd
        ) const;

private:
    QVector<Annotation> m_annotations;
};