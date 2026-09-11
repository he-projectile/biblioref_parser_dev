#!/usr/bin/env python3

import json
import re
import argparse

from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# Constants
# ============================================================

REFERENCE_LABEL = "БИБЛ. ССЫЛКА"


# ============================================================
# Regex helpers
# ============================================================

def escape_literal(character):
    return re.escape(character)


def character_class(character):
    """
    Возвращает класс символа для обобщённого regex.
    """

    if "А" <= character <= "Я" or character == "Ё":
        return "[А-ЯЁ]"

    if "а" <= character <= "я" or character == "ё":
        return "[а-яё]"

    if "A" <= character <= "Z":
        return "[A-Z]"

    if "a" <= character <= "z":
        return "[a-z]"

    if character.isdigit():
        return r"\d"

    return escape_literal(character)


def generalize_word(word):
    """
    Обобщает отдельное слово.

    Например:

        Иванов  -> [А-ЯЁ][а-яё]+
        И       -> [А-ЯЁ]
        ABC     -> [A-Z]+
        2021    -> \d+
    """

    if not word:
        return ""

    if word.isdigit():
        return r"\d+"

    # Инициалы: И, А, П и т.п.
    if len(word) == 1 and (
        "А" <= word <= "Я"
        or word == "Ё"
        or "A" <= word <= "Z"
    ):
        return "[А-ЯЁA-Z]"

    # Кириллическое слово с заглавной буквы
    if re.fullmatch(r"[А-ЯЁ][а-яё]+", word):
        return r"[А-ЯЁ][а-яё]+"

    # Кириллическое слово в нижнем регистре
    if re.fullmatch(r"[а-яё]+", word):
        return r"[а-яё]+"

    # Латинское слово
    if re.fullmatch(r"[A-Z][a-z]+", word):
        return r"[A-Z][a-z]+"

    if re.fullmatch(r"[a-z]+", word):
        return r"[a-z]+"

    # Последняя попытка — посимвольное обобщение
    result = []

    for character in word:
        result.append(character_class(character))

    return "".join(result)


def generalize_text(text):
    """
    Превращает реальный текст сущности в обобщённый regex.

    Пример:

        "Иванов И. И."

    ->

        "[А-ЯЁ][а-яё]+\\s+[А-ЯЁ]\\.\\s+[А-ЯЁ]\\."
    """

    tokens = re.findall(
        r"[А-ЯЁа-яёA-Za-z0-9]+|[^\w\s]|\s+",
        text,
        flags=re.UNICODE
    )

    result = []

    for token in tokens:

        if token.isspace():
            result.append(r"\s+")
            continue

        if re.fullmatch(
            r"[А-ЯЁа-яёA-Za-z0-9]+",
            token,
            flags=re.UNICODE
        ):
            result.append(generalize_word(token))
            continue

        result.append(re.escape(token))

    # Объединяем повторяющиеся \s+
    regex = "".join(result)

    regex = re.sub(
        r"(?:\\s\+)+",
        r"\\s+",
        regex
    )

    return regex


# ============================================================
# Micro-pattern extraction
# ============================================================

def extract_micro_patterns(text):
    """
    Извлекает атомарные признаки из текста сущности.

    В отличие от generalize_text(), здесь мы не пытаемся
    описать всю сущность целиком.
    """

    patterns = []

    # --------------------------------------------------------
    # Years
    # --------------------------------------------------------

    if re.search(
        r"(?<!\d)(17|18|19|20|21)\d{2}(?!\d)",
        text
    ):
        patterns.append(
            (
                "YEAR",
                r"(?<!\d)(17|18|19|20|21)\d{2}(?!\d)"
            )
        )

    # --------------------------------------------------------
    # Initials
    # --------------------------------------------------------

    if re.search(
        r"\b[А-ЯЁ]\.",
        text
    ):
        patterns.append(
            (
                "INITIAL",
                r"\b[А-ЯЁ]\."
            )
        )

    # --------------------------------------------------------
    # Initial pair
    # --------------------------------------------------------

    if re.search(
        r"\b[А-ЯЁ]\.\s*[А-ЯЁ]\.",
        text
    ):
        patterns.append(
            (
                "INITIALS",
                r"\b[А-ЯЁ]\.\s*[А-ЯЁ]\."
            )
        )

    # --------------------------------------------------------
    # Number
    # --------------------------------------------------------

    if re.search(
        r"\d+",
        text
    ):
        patterns.append(
            (
                "NUMBER",
                r"\d+"
            )
        )

    # --------------------------------------------------------
    # Pages
    # --------------------------------------------------------

    if re.search(
        r"\b[Сс]\.?\s*\d+(?:\s*[-–—]\s*\d+)?",
        text
    ):
        patterns.append(
            (
                "PAGES",
                r"\b[Сс]\.?\s*\d+(?:\s*[-–—]\s*\d+)?"
            )
        )

    # --------------------------------------------------------
    # DOI
    # --------------------------------------------------------

    if re.search(
        r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+",
        text
    ):
        patterns.append(
            (
                "DOI",
                r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+"
            )
        )

    # --------------------------------------------------------
    # URL
    # --------------------------------------------------------

    if re.search(
        r"(?:https?://|www\.)[^\s]+",
        text
    ):
        patterns.append(
            (
                "URL",
                r"(?:https?://|www\.)[^\s]+"
            )
        )

    # --------------------------------------------------------
    # ISBN
    # --------------------------------------------------------

    if re.search(
        r"\bISBN(?:-1[03])?\s*[:\-]?\s*[\dXx][\dXx\- ]{8,20}\b",
        text
    ):
        patterns.append(
            (
                "ISBN",
                r"\bISBN(?:-1[03])?\s*[:\-]?\s*[\dXx][\dXx\- ]{8,20}\b"
            )
        )

    # --------------------------------------------------------
    # ISSN
    # --------------------------------------------------------

    if re.search(
        r"\bISSN\s*[:\-]?\s*\d{4}[-–]\d{3}[\dXx]\b",
        text
    ):
        patterns.append(
            (
                "ISSN",
                r"\bISSN\s*[:\-]?\s*\d{4}[-–]\d{3}[\dXx]\b"
            )
        )

    # --------------------------------------------------------
    # Structural punctuation
    # --------------------------------------------------------

    structural_patterns = {
        "DOUBLE_SLASH": r"//",
        "COLON": r":",
        "SEMICOLON": r";",
        "DOT": r"\.",
        "COMMA": r",",
        "DASH": r"[-–—]",
        "SLASH": r"/",
        "PARENTHESIS": r"[()]",
        "QUOTATION": r"[«»\"„“”]",
    }

    for name, regex in structural_patterns.items():

        if re.search(regex, text):
            patterns.append(
                (name, regex)
            )

    return patterns


# ============================================================
# Separator normalization
# ============================================================

def normalize_separator(separator):
    """
    Нормализует текст между двумя сущностями.

    Например:

        "   //  "
        " // "
        "//"

    превращаются в один структурный шаблон.
    """

    if not separator:
        return ""

    # Пробельные последовательности
    separator = re.sub(
        r"\s+",
        " ",
        separator
    ).strip()

    # Полностью удаляем окружающие пробелы.
    # Сам факт наличия пробела будет представлен \s+.
    separator = separator.strip()

    if not separator:
        return r"\s+"

    result = []

    for character in separator:

        if character.isspace():
            result.append(r"\s+")
        else:
            result.append(
                re.escape(character)
            )

    regex = "".join(result)

    regex = re.sub(
        r"(?:\\s\+)+",
        r"\\s+",
        regex
    )

    return regex


# ============================================================
# Annotation loading
# ============================================================

def load_annotations(json_path):
    """
    Загружает annotations из JSON.

    Поддерживается структура:

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
    """

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return data.get(
        "annotations",
        []
    )


# ============================================================
# Reference extraction
# ============================================================

def get_references(annotations):
    return [
        annotation
        for annotation in annotations
        if annotation.get("label") == REFERENCE_LABEL
    ]


def get_child_entities(reference):
    """
    Рекурсивно извлекает все дочерние сущности
    библиографической ссылки.
    """

    children = []

    def walk(nodes):

        for node in nodes:

            if not isinstance(node, dict):
                continue

            # Если у узла есть children — сначала обходим их
            nested = node.get("children", [])

            if nested:
                walk(nested)

            # Сам узел является сущностью
            if (
                node.get("label")
                and node.get("start") is not None
                and node.get("end") is not None
            ):
                children.append(node)

    walk(
        reference.get("children", [])
    )

    children.sort(
        key=lambda annotation: (
            annotation["start"],
            annotation["end"]
        )
    )

    return children


# ============================================================
# Mining
# ============================================================

def mine_dataset(dataset_path):
    """
    Анализирует весь датасет.

    Ожидается:

        dataset/
            doc1.txt
            doc1.json
            doc2.txt
            doc2.json
            ...
    """

    dataset_path = Path(dataset_path)

    entity_pattern_occurrences = Counter()
    entity_pattern_documents = defaultdict(set)

    entity_type_counts = Counter()

    separator_occurrences = Counter()
    separator_documents = defaultdict(set)

    transition_counts = Counter()

    entity_regex_occurrences = Counter()
    entity_regex_documents = defaultdict(set)

    total_references = 0
    total_entities = 0
    total_transitions = 0

    text_files = sorted(
        dataset_path.glob("*.txt")
    )

    for text_path in text_files:

        json_path = text_path.with_suffix(
            ".json"
        )

        if not json_path.exists():
            print(
                f"[WARNING] Нет JSON для {text_path}"
            )
            continue

        text = text_path.read_text(
            encoding="utf-8",
            errors="replace"
        )

        annotations = load_annotations(
            json_path
        )

        references = get_references(
            annotations
        )

        for reference in references:

            total_references += 1

            children = get_child_entities(
                reference
            )

            total_entities += len(children)

            # ------------------------------------------------
            # Entities
            # ------------------------------------------------

            for entity in children:

                label = entity.get(
                    "label",
                    "UNKNOWN"
                )

                entity_text = entity.get(
                    "text"
                )

                if entity_text is None:

                    entity_text = text[
                        entity["start"]:
                        entity["end"]
                    ]

                entity_type_counts[
                    label
                ] += 1

                # Атомарные микропаттерны
                micro_patterns = extract_micro_patterns(
                    entity_text
                )

                for pattern_name, regex in micro_patterns:

                    key = (
                        label,
                        pattern_name,
                        regex
                    )

                    entity_pattern_occurrences[
                        key
                    ] += 1

                    entity_pattern_documents[
                        key
                    ].add(text_path.name)

                # Обобщённая форма сущности
                generalized_regex = generalize_text(
                    entity_text
                )

                regex_key = (
                    label,
                    generalized_regex
                )

                entity_regex_occurrences[
                    regex_key
                ] += 1

                entity_regex_documents[
                    regex_key
                ].add(text_path.name)

            # ------------------------------------------------
            # Transitions
            # ------------------------------------------------

            for current_entity, next_entity in zip(
                children,
                children[1:]
            ):

                current_end = current_entity["end"]
                next_start = next_entity["start"]

                if next_start < current_end:
                    continue

                separator = text[
                    current_end:next_start
                ]

                normalized_separator = normalize_separator(
                    separator
                )

                if not normalized_separator:
                    continue

                current_label = current_entity.get(
                    "label",
                    "UNKNOWN"
                )

                next_label = next_entity.get(
                    "label",
                    "UNKNOWN"
                )

                transition = (
                    current_label,
                    next_label
                )

                transition_counts[
                    transition
                ] += 1

                separator_key = (
                    current_label,
                    next_label,
                    normalized_separator
                )

                separator_occurrences[
                    separator_key
                ] += 1

                separator_documents[
                    separator_key
                ].add(text_path.name)

                total_transitions += 1

    return {
        "total_references": total_references,
        "total_entities": total_entities,
        "total_transitions": total_transitions,

        "entity_pattern_occurrences":
            entity_pattern_occurrences,

        "entity_pattern_documents":
            entity_pattern_documents,

        "entity_type_counts":
            entity_type_counts,

        "separator_occurrences":
            separator_occurrences,

        "separator_documents":
            separator_documents,

        "transition_counts":
            transition_counts,

        "entity_regex_occurrences":
            entity_regex_occurrences,

        "entity_regex_documents":
            entity_regex_documents,
    }


# ============================================================
# Build output
# ============================================================

def build_pattern_file(
    mining_result,
    min_support=3,
    min_documents=1
):
    """
    Формирует patterns.json.
    """

    patterns = []

    total_references = mining_result[
        "total_references"
    ]

    total_entities = mining_result[
        "total_entities"
    ]

    total_transitions = mining_result[
        "total_transitions"
    ]

    # ========================================================
    # Entity micro-patterns
    # ========================================================

    for (
        entity_label,
        pattern_name,
        regex
    ), occurrence_count in sorted(
        mining_result[
            "entity_pattern_occurrences"
        ].items(),
        key=lambda item: -item[1]
    ):

        documents = mining_result[
            "entity_pattern_documents"
        ][
            (
                entity_label,
                pattern_name,
                regex
            )
        ]

        if occurrence_count < min_support:
            continue

        if len(documents) < min_documents:
            continue

        patterns.append({
            "name": pattern_name,
            "type": "entity_micro",
            "entity": entity_label,
            "regex": regex,
            "occurrences": occurrence_count,
            "documents": len(documents),
            "support": (
                occurrence_count / max(
                    total_entities,
                    1
                )
            ),
            "weight": 1.0
        })

    # ========================================================
    # Generalized entity structures
    # ========================================================

    for (
        entity_label,
        regex
    ), occurrence_count in sorted(
        mining_result[
            "entity_regex_occurrences"
        ].items(),
        key=lambda item: -item[1]
    ):

        documents = mining_result[
            "entity_regex_documents"
        ][
            (
                entity_label,
                regex
            )
        ]

        if occurrence_count < min_support:
            continue

        if len(documents) < min_documents:
            continue

        patterns.append({
            "name": "ENTITY_SHAPE",
            "type": "entity_shape",
            "entity": entity_label,
            "regex": regex,
            "occurrences": occurrence_count,
            "documents": len(documents),
            "support": (
                occurrence_count / max(
                    total_entities,
                    1
                )
            ),
            "weight": 1.0
        })

    # ========================================================
    # Separators
    # ========================================================

    for (
        previous_label,
        next_label,
        regex
    ), occurrence_count in sorted(
        mining_result[
            "separator_occurrences"
        ].items(),
        key=lambda item: -item[1]
    ):

        documents = mining_result[
            "separator_documents"
        ][
            (
                previous_label,
                next_label,
                regex
            )
        ]

        if occurrence_count < min_support:
            continue

        if len(documents) < min_documents:
            continue

        patterns.append({
            "name": "SEPARATOR",
            "type": "transition",
            "from": previous_label,
            "to": next_label,
            "regex": regex,
            "occurrences": occurrence_count,
            "documents": len(documents),
            "support": (
                occurrence_count / max(
                    total_transitions,
                    1
                )
            ),
            "weight": 1.0
        })

    # ========================================================
    # Metadata
    # ========================================================

    return {
        "version": 1,

        "statistics": {
            "references":
                total_references,

            "entities":
                total_entities,

            "transitions":
                total_transitions,

            "entity_types": dict(
                mining_result[
                    "entity_type_counts"
                ]
            )
        },

        "patterns": patterns
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Автоматическое построение библиографических "
            "микропаттернов из размеченного датасета"
        )
    )

    parser.add_argument(
        "dataset",
        help=(
            "Папка с парами *.txt + *.json"
        )
    )

    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help=(
            "Куда сохранить patterns.json"
        )
    )

    parser.add_argument(
        "--min-support",
        type=int,
        default=3,
        help=(
            "Минимальное число наблюдений паттерна"
        )
    )

    parser.add_argument(
        "--min-documents",
        type=int,
        default=1,
        help=(
            "Минимальное количество документов"
        )
    )

    args = parser.parse_args()

    mining_result = mine_dataset(
        args.dataset
    )

    pattern_file = build_pattern_file(
        mining_result,
        min_support=args.min_support,
        min_documents=args.min_documents
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            pattern_file,
            file,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PATTERN MINER")
    print("=" * 70)

    print(
        f"References:  "
        f"{pattern_file['statistics']['references']}"
    )

    print(
        f"Entities:    "
        f"{pattern_file['statistics']['entities']}"
    )

    print(
        f"Transitions: "
        f"{pattern_file['statistics']['transitions']}"
    )

    print(
        f"Patterns:    "
        f"{len(pattern_file['patterns'])}"
    )

    print()
    print(
        f"Результат сохранён: {output_path}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()