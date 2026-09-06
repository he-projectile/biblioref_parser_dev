from __future__ import annotations

from dataclasses import dataclass

from SoftWare.PythonWorkflow.NeuroSlop.localization.loader import Annotation, Document


@dataclass(frozen=True)
class Line:
    index: int
    start: int
    end: int
    text: str

    @property
    def length(self) -> int:
        return self.end - self.start


def split_lines(text: str) -> list[Line]:
    """
    Разбивает текст на строки, сохраняя абсолютные
    позиции символов в исходном документе.

    end — исключительная граница:
        text[start:end]
    """

    lines = []

    offset = 0

    for index, raw_line in enumerate(text.splitlines(keepends=True)):
        line_without_newline = raw_line.rstrip("\r\n")

        start = offset
        end = start + len(line_without_newline)

        lines.append(
            Line(
                index=index,
                start=start,
                end=end,
                text=line_without_newline,
            )
        )

        offset += len(raw_line)

    # splitlines() не создаёт строку для пустого текста
    # и не всегда удобно обрабатывает последний пустой фрагмент.
    if not lines and text:
        lines.append(
            Line(
                index=0,
                start=0,
                end=len(text),
                text=text,
            )
        )

    return lines


def intervals_overlap(
    start_a: int,
    end_a: int,
    start_b: int,
    end_b: int,
) -> bool:
    """
    Проверяет пересечение полуинтервалов:

        [start_a, end_a)
        [start_b, end_b)
    """

    return (
        start_a < end_b
        and end_a > start_b
    )


def line_contains_annotation(
    line: Line,
    annotation: Annotation,
) -> bool:
    return intervals_overlap(
        line.start,
        line.end,
        annotation.start,
        annotation.end,
    )


def label_lines(
    document: Document,
    lines: list[Line],
) -> list[tuple[Line, bool]]:
    """
    Помечает строки, пересекающиеся с БИБЛ. ССЫЛКА.

    True  -> строка содержит часть библиографической ссылки
    False -> не содержит.
    """

    references = document.references

    result = []

    for line in lines:
        positive = any(
            line_contains_annotation(line, reference)
            for reference in references
        )

        result.append((line, positive))

    return result