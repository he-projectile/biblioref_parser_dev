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

    Annotation newAnnotation = annotation;

    // ------------------------------------------------------------
    // Определяем родителя автоматически.
    //
    // Если новая сущность находится внутри существующей,
    // она становится её дочерней сущностью.
    // ------------------------------------------------------------

    newAnnotation.parentIndex =
        findParent(
            annotation.start,
            annotation.end
            );

    if (newAnnotation.parentIndex >= 0)
    {
        const Annotation& parent =
            m_annotations[newAnnotation.parentIndex];

        if (parent.label != "БИБЛ. ССЫЛКА")
        {
            return false;
        }
    }

    // ------------------------------------------------------------
    // Проверяем пересечения.
    //
    // Полностью содержащая сущность допустима:
    //
    // БИБЛ. ССЫЛКА
    // ├── TITLE
    // └── YEAR
    //
    // Частичное пересечение запрещено:
    //
    // ┌───────────────┐
    //       ┌───────────────┐
    //       └───────────────┘
    // └───────────────┘
    //
    // ------------------------------------------------------------

    for (int i = 0; i < m_annotations.size(); ++i)
    {
        const Annotation& existing =
            m_annotations[i];

        if (rangesOverlap(
                annotation.start,
                annotation.end,
                existing.start,
                existing.end))
        {
            const bool newContainsOld =
                rangeContains(
                    annotation.start,
                    annotation.end,
                    existing.start,
                    existing.end
                    );

            const bool oldContainsNew =
                rangeContains(
                    existing.start,
                    existing.end,
                    annotation.start,
                    annotation.end
                    );

            // Полное вложение допустимо.
            if (newContainsOld || oldContainsNew)
                continue;

            // Частичное пересечение запрещено.
            return false;
        }
    }

    m_annotations.append(newAnnotation);

    return true;
}

bool AnnotationModel::removeAnnotation(int index)
{
    if (index < 0 ||
        index >= m_annotations.size())
    {
        return false;
    }

    m_annotations.removeAt(index);

    // После удаления индексы родителей могут измениться.
    //
    // Пересчитываем их заново.
    for (int i = 0; i < m_annotations.size(); ++i)
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