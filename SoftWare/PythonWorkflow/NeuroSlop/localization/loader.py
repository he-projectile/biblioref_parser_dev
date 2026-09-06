from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REFERENCE_LABEL = "БИБЛ. ССЫЛКА"


@dataclass(frozen=True)
class Annotation:
    start: int
    end: int
    label: str
    text: str

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class Document:
    name: str
    text_path: Path
    annotation_path: Path
    text: str
    annotations: list[Annotation]

    @property
    def length(self) -> int:
        return len(self.text)

    @property
    def references(self) -> list[Annotation]:
        return [
            annotation
            for annotation in self.annotations
            if annotation.label == REFERENCE_LABEL
        ]


class DatasetLoader:
    """
    Загружает пары:
        document.txt
        document.json

    JSON должен содержать:
        {
            "annotations": [
                {
                    "start": ...,
                    "end": ...,
                    "label": ...,
                    "text": ...
                }
            ]
        }

    Поле children игнорируется, поскольку для localization
    оно не требуется.
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"DataSource не существует: {self.data_dir}"
            )

        if not self.data_dir.is_dir():
            raise NotADirectoryError(
                f"Это не директория: {self.data_dir}"
            )

    def load_all(self) -> list[Document]:
        documents = []

        for txt_path in sorted(self.data_dir.glob("*.txt")):
            json_path = txt_path.with_suffix(".json")

            if not json_path.exists():
                print(
                    f"[WARNING] Нет JSON для {txt_path.name}: "
                    f"{json_path.name}"
                )
                continue

            try:
                document = self.load(txt_path, json_path)
                documents.append(document)
            except Exception as exc:
                print(
                    f"[ERROR] Не удалось загрузить "
                    f"{txt_path.name}: {exc}"
                )

        return documents

    def load(
        self,
        txt_path: str | Path,
        json_path: str | Path,
    ) -> Document:

        txt_path = Path(txt_path)
        json_path = Path(json_path)

        text = txt_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        with json_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data: Any = json.load(file)

        raw_annotations = data.get("annotations", [])

        annotations = []

        for index, raw in enumerate(raw_annotations):
            try:
                start = int(raw["start"])
                end = int(raw["end"])
                label = str(raw["label"])
                annotation_text = str(raw.get("text", ""))

                annotations.append(
                    Annotation(
                        start=start,
                        end=end,
                        label=label,
                        text=annotation_text,
                    )
                )

            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Некорректная annotation #{index} "
                    f"в {json_path.name}: {exc}"
                ) from exc

        annotations.sort(key=lambda x: (x.start, x.end))

        return Document(
            name=txt_path.stem,
            text_path=txt_path,
            annotation_path=json_path,
            text=text,
            annotations=annotations,
        )