#pragma once

#include <QString>

struct Annotation
{
    int start = 0;
    int end = 0;       // [start, end)

    QString label;
    QString text;

    // Индекс родительской сущности.
    // -1 означает отсутствие родителя.
    int parentIndex = -1;
};