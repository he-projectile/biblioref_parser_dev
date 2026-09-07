#include "AnnotationModel.h"
#include <climits>

void AnnotationModel::clear()
{
    m_annotations.clear();
}

const QVector<Annotation>& AnnotationModel::annotations() const
{
    return m_annotations;
}

bool AnnotationModel::addAnnotation(
    const Annotation& annotation
    )
{
    if (annotation.start < 0)
        return false;

    if (annotation.end <= annotation.start)
        return false;

    if (annotation.label.isEmpty())
        return false;

    // ------------------------------------------------------------
    // REFERENCE
    // ------------------------------------------------------------

    if (annotation.label ==
        "БИБЛ. ССЫЛКА")
    {
        // REFERENCE не может пересекаться
        // вообще ни с чем.
        if (hasAnyOverlap(
                annotation.start,
                annotation.end))
        {
            return false;
        }
    }

    // ------------------------------------------------------------
    // Дочерняя сущность
    // ------------------------------------------------------------

    else
    {
        // Должна находиться внутри REFERENCE.
        if (!isInsideReference(
                annotation.start,
                annotation.end))
        {
            return false;
        }

        // Дочерние сущности не пересекаются.
        if (hasChildOverlap(
                annotation.start,
                annotation.end))
        {
            return false;
        }
    }

    Annotation newAnnotation =
        annotation;

    newAnnotation.parentIndex =
        findParent(
            annotation.start,
            annotation.end
            );

    m_annotations.append(
        newAnnotation
        );

    return true;
}

bool AnnotationModel::removeAnnotation(
    int index
    )
{
    if (index < 0 ||
        index >= m_annotations.size())
    {
        return false;
    }

    const Annotation& annotation =
        m_annotations[index];

    // ------------------------------------------------------------
    // Если удаляем REFERENCE,
    // удаляем всю её ветку.
    // ------------------------------------------------------------

    if (annotation.label ==
        "БИБЛ. ССЫЛКА")
    {
        const int referenceStart =
            annotation.start;

        const int referenceEnd =
            annotation.end;

        for (int i =
             m_annotations.size() - 1;
             i >= 0;
             --i)
        {
            const Annotation& current =
                m_annotations[i];

            if (current.start >= referenceStart &&
                current.end <= referenceEnd)
            {
                m_annotations.removeAt(i);
            }
        }
    }
    else
    {
        // Обычная сущность —
        // удаляем только её.
        m_annotations.removeAt(index);
    }

    // ------------------------------------------------------------
    // Пересчитываем parentIndex
    // ------------------------------------------------------------

    for (int i = 0;
         i < m_annotations.size();
         ++i)
    {
        m_annotations[i].parentIndex =
            findParent(
                m_annotations[i].start,
                m_annotations[i].end
                );
    }

    return true;
}

bool AnnotationModel::updateLabel(
    int index,
    const QString& label
    )
{
    if (index < 0 ||
        index >= m_annotations.size())
    {
        return false;
    }

    if (label.isEmpty())
        return false;

    m_annotations[index].label = label;

    return true;
}

int AnnotationModel::annotationAt(int position) const
{
    int bestIndex = -1;

    int bestLength = INT_MAX;

    for (int i = 0; i < m_annotations.size(); ++i)
    {
        const Annotation& annotation =
            m_annotations[i];

        if (position >= annotation.start &&
            position < annotation.end)
        {
            const int length =
                annotation.end -
                annotation.start;

            // Если есть вложенные сущности,
            // выбираем самую глубокую/маленькую.
            if (length < bestLength)
            {
                bestLength = length;
                bestIndex = i;
            }
        }
    }

    return bestIndex;
}

bool AnnotationModel::hasOverlap(
    int start,
    int end,
    int ignoreIndex
    ) const
{
    for (int i = 0; i < m_annotations.size(); ++i)
    {
        if (i == ignoreIndex)
            continue;

        const Annotation& annotation =
            m_annotations[i];

        if (!rangesOverlap(
                start,
                end,
                annotation.start,
                annotation.end))
        {
            continue;
        }

        const bool firstContainsSecond =
            rangeContains(
                start,
                end,
                annotation.start,
                annotation.end
                );

        const bool secondContainsFirst =
            rangeContains(
                annotation.start,
                annotation.end,
                start,
                end
                );

        // Вложенность допустима.
        if (firstContainsSecond ||
            secondContainsFirst)
        {
            continue;
        }

        // Частичное пересечение.
        return true;
    }

    return false;
}

int AnnotationModel::findParent(
    int start,
    int end
    ) const
{
    int parentIndex = -1;

    int parentLength = INT_MAX;

    for (int i = 0; i < m_annotations.size(); ++i)
    {
        const Annotation& annotation =
            m_annotations[i];

        if (!rangeContains(
                annotation.start,
                annotation.end,
                start,
                end))
        {
            continue;
        }

        const int length =
            annotation.end -
            annotation.start;

        // Выбираем ближайшего родителя,
        // а не самого большого предка.
        if (length < parentLength)
        {
            parentLength = length;
            parentIndex = i;
        }
    }

    return parentIndex;
}

bool AnnotationModel::rangesOverlap(
    int start1,
    int end1,
    int start2,
    int end2
    ) const
{
    return start1 < end2 &&
           end1 > start2;
}

bool AnnotationModel::rangeContains(
    int outerStart,
    int outerEnd,
    int innerStart,
    int innerEnd
    ) const
{
    return outerStart <= innerStart &&
           outerEnd >= innerEnd;
}

bool AnnotationModel::hasAnyOverlap(
    int start,
    int end
    ) const
{
    for (const Annotation& annotation :
         m_annotations)
    {
        if (rangesOverlap(
                start,
                end,
                annotation.start,
                annotation.end))
        {
            return true;
        }
    }

    return false;
}

bool AnnotationModel::hasChildOverlap(
    int start,
    int end
    ) const
{
    for (const Annotation& annotation :
         m_annotations)
    {
        if (annotation.label ==
            "БИБЛ. ССЫЛКА")
        {
            continue;
        }

        if (rangesOverlap(
                start,
                end,
                annotation.start,
                annotation.end))
        {
            return true;
        }
    }

    return false;
}

bool AnnotationModel::isInsideReference(
    int start,
    int end
    ) const
{
    for (const Annotation& annotation :
         m_annotations)
    {
        if (annotation.label !=
            "БИБЛ. ССЫЛКА")
        {
            continue;
        }

        if (rangeContains(
                annotation.start,
                annotation.end,
                start,
                end))
        {
            return true;
        }
    }

    return false;
}