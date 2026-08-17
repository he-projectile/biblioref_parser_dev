#pragma once

#include "Annotation.h"

#include <QString>
#include <QVector>

namespace JsonStorage
{
    bool save(
        const QString& filePath,
        const QString& documentName,
        const QVector<Annotation>& annotations
    );

    bool load(
        const QString& filePath,
        QString& documentName,
        QVector<Annotation>& annotations,
        QString* errorMessage = nullptr
    );
}