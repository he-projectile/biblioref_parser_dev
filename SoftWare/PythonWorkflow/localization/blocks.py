from __future__ import annotations

from dataclasses import dataclass

from loader import Annotation, Document


@dataclass
class ReferenceBlock:
    start: int
    end: int
    references: list[Annotation]

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def reference_count(self) -> int:
        return len(self.references)


def group_references(
    document: Document,
    max_gap: int = 200,
) -> list[ReferenceBlock]:
    """
    Группирует БИБЛ. ССЫЛКА в блоки.

    max_gap:
        максимальное количество символов между соседними
        библиографическими ссылками, при котором они всё ещё
        считаются принадлежащими одному блоку.

    ВАЖНО:
        это пока эвристика для анализа корпуса.
        На этом этапе она НЕ является финальным алгоритмом
        localization.
    """

    references = document.references

    if not references:
        return []

    blocks: list[ReferenceBlock] = []

    current_references = [references[0]]

    for reference in references[1:]:
        previous = current_references[-1]

        gap = reference.start - previous.end

        if gap <= max_gap:
            current_references.append(reference)
        else:
            blocks.append(
                ReferenceBlock(
                    start=current_references[0].start,
                    end=current_references[-1].end,
                    references=current_references,
                )
            )

            current_references = [reference]

    blocks.append(
        ReferenceBlock(
            start=current_references[0].start,
            end=current_references[-1].end,
            references=current_references,
        )
    )

    return blocks