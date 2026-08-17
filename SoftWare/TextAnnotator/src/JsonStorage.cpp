#include "JsonStorage.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include <functional>

namespace JsonStorage
{

bool save(
    const QString& filePath,
    const QString& documentName,
    const QVector<Annotation>& annotations
    )
{
    // ------------------------------------------------------------
    // Рекурсивное преобразование Annotation -> QJsonObject
    // ------------------------------------------------------------

    std::function<QJsonObject(int)> annotationToJson;

    annotationToJson =
        [&annotations, &annotationToJson](int index) -> QJsonObject
    {
        const Annotation& annotation =
            annotations[index];

        QJsonObject object;

        object["start"] = annotation.start;
        object["end"] = annotation.end;
        object["label"] = annotation.label;
        object["text"] = annotation.text;

        // --------------------------------------------------------
        // Ищем непосредственных детей.
        //
        // Ребёнок должен полностью находиться внутри текущей
        // сущности.
        // --------------------------------------------------------

        QJsonArray children;

        for (int i = 0; i < annotations.size(); ++i)
        {
            if (i == index)
                continue;

            const Annotation& candidate =
                annotations[i];

            // candidate находится внутри annotation
            if (candidate.start >= annotation.start &&
                candidate.end <= annotation.end)
            {
                // Проверяем, что это именно непосредственный
                // ребёнок, а не более глубокий потомок.
                bool isDirectChild = true;

                for (int j = 0; j < annotations.size(); ++j)
                {
                    if (j == index || j == i)
                        continue;

                    const Annotation& possibleParent =
                        annotations[j];

                    const bool containsCandidate =
                        possibleParent.start <= candidate.start &&
                        possibleParent.end >= candidate.end;

                    const bool insideAnnotation =
                        possibleParent.start >= annotation.start &&
                        possibleParent.end <= annotation.end;

                    if (containsCandidate &&
                        insideAnnotation)
                    {
                        // possibleParent находится между
                        // annotation и candidate.
                        isDirectChild = false;
                        break;
                    }
                }

                if (isDirectChild)
                {
                    children.append(
                        annotationToJson(i)
                        );
                }
            }
        }

        if (!children.isEmpty())
        {
            object["children"] = children;
        }

        return object;
    };

    // ------------------------------------------------------------
    // Корневой JSON
    // ------------------------------------------------------------

    QJsonObject root;

    root["document"] = documentName;

    QJsonArray jsonAnnotations;

    // ------------------------------------------------------------
    // В корень помещаем только сущности, у которых нет
    // содержащей сущности.
    //
    // Поэтому REFERENCE окажется здесь,
    // а AUTHOR/TITLE/JOURNAL/etc. — внутри него.
    // ------------------------------------------------------------

    for (int i = 0; i < annotations.size(); ++i)
    {
        const Annotation& annotation =
            annotations[i];

        bool hasParent = false;

        for (int j = 0; j < annotations.size(); ++j)
        {
            if (i == j)
                continue;

            const Annotation& possibleParent =
                annotations[j];

            if (possibleParent.start <= annotation.start &&
                possibleParent.end >= annotation.end)
            {
                // Исключаем случай одинаковых диапазонов.
                if (possibleParent.start == annotation.start &&
                    possibleParent.end == annotation.end)
                {
                    continue;
                }

                hasParent = true;
                break;
            }
        }

        if (!hasParent)
        {
            jsonAnnotations.append(
                annotationToJson(i)
                );
        }
    }

    root["annotations"] = jsonAnnotations;

    // ------------------------------------------------------------
    // Запись файла
    // ------------------------------------------------------------

    QJsonDocument document(root);

    QFile file(filePath);

    if (!file.open(QIODevice::WriteOnly))
        return false;

    const qint64 bytesWritten =
        file.write(
            document.toJson(
                QJsonDocument::Indented
                )
            );

    file.close();

    return bytesWritten >= 0;
}

bool load(
    const QString& filePath,
    QString& documentName,
    QVector<Annotation>& annotations,
    QString* errorMessage
    )
{
    QFile file(filePath);

    if (!file.open(QIODevice::ReadOnly))
        return false;

    const QByteArray data =
        file.readAll();

    file.close();

    QJsonParseError error;

    const QJsonDocument document =
        QJsonDocument::fromJson(
            data,
            &error
            );

    if (error.error != QJsonParseError::NoError)
        return false;

    if (!document.isObject())
        return false;

    const QJsonObject root =
        document.object();

    // ------------------------------------------------------------
    // Имя документа
    // ------------------------------------------------------------

    documentName =
        root["document"].toString();

    annotations.clear();

    // ------------------------------------------------------------
    // Рекурсивная загрузка сущностей.
    //
    // В JSON:
    //
    // REFERENCE
    //   ├── AUTHOR
    //   ├── TITLE
    //   └── YEAR
    //
    // В QVector:
    //
    // REFERENCE
    // AUTHOR
    // TITLE
    // YEAR
    //
    // parentIndex при этом НЕ нужен.
    // Иерархия определяется диапазонами.
    // ------------------------------------------------------------

    std::function<void(const QJsonObject&)> loadAnnotation;

    loadAnnotation =
        [&annotations, &loadAnnotation](
            const QJsonObject& object
            )
    {
        Annotation annotation;

        annotation.start =
            object["start"].toInt();

        annotation.end =
            object["end"].toInt();

        annotation.label =
            object["label"].toString();

        annotation.text =
            object["text"].toString();

        annotation.parentIndex = -1;

        annotations.append(annotation);

        // --------------------------------------------------------
        // Загружаем дочерние сущности
        // --------------------------------------------------------

        const QJsonArray children =
            object["children"].toArray();

        for (const QJsonValue& child :
             children)
        {
            if (!child.isObject())
                continue;

            loadAnnotation(
                child.toObject()
                );
        }
    };

    // ------------------------------------------------------------
    // Корневые сущности
    // ------------------------------------------------------------

    const QJsonArray jsonAnnotations =
        root["annotations"].toArray();

    for (const QJsonValue& value :
         jsonAnnotations)
    {
        if (!value.isObject())
            continue;

        loadAnnotation(
            value.toObject()
            );
    }

    return true;
}

}