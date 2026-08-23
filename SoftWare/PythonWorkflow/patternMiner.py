import argparse
import json
import re
from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# НАСТРОЙКИ
# ============================================================

REFERENCE_LABEL = "БИБЛ. ССЫЛКА"

# Поля, для которых имеет смысл автоматически искать regex.
FIELD_REGEX = {
    "ГОД ПУБЛИКАЦИИ": [
        r"(?<!\d)(?:19|20)\d{2}(?!\d)"
    ],

    "СТРАНИЦЫ": [
        r"\b\d{1,5}\s*[-–—]\s*\d{1,5}\b",
        r"\b[Ss]\.?\s*\d+(?:\s*[-–—]\s*\d+)?",
        r"\b[Pp]\.?\s*\d+(?:\s*[-–—]\s*\d+)?",
    ],

    "DOI": [
        r"10\.\d{4,9}/[-._;()/:A-Z0-9]+"
    ],

    "URL": [
        r"https?://[^\s]+"
    ],

    "ISBN": [
        r"\bISBN(?:[- ]?(?:13|10))?[- ]?"
        r"(?:97[89][- ]?)?"
        r"[\dXx][\dXx -]{8,20}"
    ],

    "ISSN": [
        r"\b\d{4}[-–]\d{3}[\dXx]\b"
    ],
}


# ============================================================
# JSON
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


# ============================================================
# НОРМАЛИЗАЦИЯ ПРОБЕЛОВ
# ============================================================

def normalize_spaces(text):

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# НОРМАЛИЗАЦИЯ РАЗДЕЛИТЕЛЕЙ
# ============================================================

def normalize_separator(text):

    if not text:
        return ""

    # Переносы строк превращаем в пробел
    text = text.replace(
        "\r",
        " "
    )

    text = text.replace(
        "\n",
        " "
    )

    # Табуляция
    text = text.replace(
        "\t",
        " "
    )

    # Несколько пробелов
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# ПОЛЯ
# ============================================================

def get_children(annotation):

    children = annotation.get(
        "children",
        []
    )

    children = [
        child
        for child in children
        if "start" in child
        and "end" in child
        and "label" in child
    ]

    children.sort(
        key=lambda x: (
            int(x["start"]),
            int(x["end"])
        )
    )

    return children


# ============================================================
# ПРОВЕРКА REGEX
# ============================================================

def regex_match_field(
    label,
    text
):

    patterns = FIELD_REGEX.get(
        label,
        []
    )

    if not patterns:
        return []

    matches = []

    for pattern in patterns:

        try:

            regex = re.compile(
                pattern,
                re.IGNORECASE
            )

        except re.error:
            continue

        for match in regex.finditer(
            text
        ):

            matches.append(
                {
                    "pattern": pattern,
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end()
                }
            )

    return matches


# ============================================================
# СТАТИСТИКА ПОЛЯ
# ============================================================

def analyze_field(
    child,
    reference_text
):

    label = child["label"]

    text = child.get(
        "text",
        ""
    )

    text = normalize_spaces(
        text
    )

    length = len(text)

    regex_matches = regex_match_field(
        label,
        text
    )

    regex_covers_whole_field = False

    for match in regex_matches:

        if (
            match["start"] == 0
            and
            match["end"] == len(text)
        ):
            regex_covers_whole_field = True
            break

    return {
        "label": label,

        "count": 1,

        "length": length,

        "text_example": text,

        "regex_matches": len(
            regex_matches
        ),

        "regex_covers_whole_field":
            regex_covers_whole_field,
    }


# ============================================================
# РАЗДЕЛИТЕЛЬ МЕЖДУ ПОЛЯМИ
# ============================================================

def extract_separator(
    first,
    second,
    reference_start,
    reference_text
):

    first_end = int(
        first["end"]
    )

    second_start = int(
        second["start"]
    )

    # Переводим глобальные позиции
    # в локальные позиции ссылки.

    first_end_local = (
        first_end
        - reference_start
    )

    second_start_local = (
        second_start
        - reference_start
    )

    if (
        first_end_local < 0
        or second_start_local < 0
        or first_end_local > len(reference_text)
        or second_start_local > len(reference_text)
    ):
        return ""

    separator = reference_text[
        first_end_local:
        second_start_local
    ]

    return normalize_separator(
        separator
    )


# ============================================================
# ПАТТЕРН ССЫЛКИ
# ============================================================

def get_pattern(children):

    return tuple(
        child["label"]
        for child in children
    )


# ============================================================
# АНАЛИЗ ОДНОЙ ССЫЛКИ
# ============================================================

def analyze_reference(
    reference,
    children
):

    reference_start = int(
        reference["start"]
    )

    reference_end = int(
        reference["end"]
    )

    reference_text = reference.get(
        "text",
        ""
    )

    # --------------------------------------------------------
    # Если текст ссылки присутствует в JSON,
    # используем его.
    # --------------------------------------------------------

    if not reference_text:

        reference_text = ""

    # --------------------------------------------------------
    # Структура
    # --------------------------------------------------------

    pattern = get_pattern(
        children
    )

    # --------------------------------------------------------
    # Поля
    # --------------------------------------------------------

    fields = []

    for child in children:

        fields.append(
            analyze_field(
                child,
                reference_text
            )
        )

    # --------------------------------------------------------
    # Переходы
    # --------------------------------------------------------

    transitions = []

    for i in range(
        len(children) - 1
    ):

        first = children[i]

        second = children[i + 1]

        separator = extract_separator(
            first,
            second,
            reference_start,
            reference_text
        )

        transitions.append(
            {
                "from": first["label"],
                "to": second["label"],
                "separator": separator
            }
        )

    return {
        "pattern": pattern,

        "reference": {
            "start": reference_start,
            "end": reference_end,
            "length": (
                reference_end
                - reference_start
            )
        },

        "fields": fields,

        "transitions": transitions
    }


# ============================================================
# АГРЕГАЦИЯ PATTERN
# ============================================================

def aggregate_patterns(
    references
):

    patterns = {}

    for item in references:

        pattern = tuple(
            item["pattern"]
        )

        if pattern not in patterns:

            patterns[pattern] = {
                "pattern": list(pattern),

                "frequency": 0,

                "field_statistics":
                    defaultdict(
                        lambda: {
                            "count": 0,
                            "lengths": [],
                            "regex_matches": 0,
                            "regex_full_matches": 0,
                            "examples": []
                        }
                    ),

                "transitions":
                    defaultdict(
                        lambda: {
                            "count": 0,
                            "separators":
                                Counter()
                        }
                    ),

                "reference_lengths": []
            }

        data = patterns[
            pattern
        ]

        data["frequency"] += 1

        # ----------------------------------------------------
        # Длина ссылки
        # ----------------------------------------------------

        data[
            "reference_lengths"
        ].append(
            item["reference"]["length"]
        )

        # ----------------------------------------------------
        # Поля
        # ----------------------------------------------------

        for field in item["fields"]:

            label = field[
                "label"
            ]

            stats = data[
                "field_statistics"
            ][label]

            stats["count"] += 1

            stats[
                "lengths"
            ].append(
                field["length"]
            )

            stats[
                "regex_matches"
            ] += (
                1
                if field["regex_matches"] > 0
                else 0
            )

            stats[
                "regex_full_matches"
            ] += (
                1
                if field[
                    "regex_covers_whole_field"
                ]
                else 0
            )

            if (
                len(stats["examples"])
                < 10
            ):

                example = (
                    field["text_example"]
                )

                if example:

                    stats[
                        "examples"
                    ].append(
                        example
                    )

        # ----------------------------------------------------
        # Переходы
        # ----------------------------------------------------

        for transition in (
            item["transitions"]
        ):

            key = (
                transition["from"],
                transition["to"]
            )

            transition_stats = (
                data["transitions"][key]
            )

            transition_stats[
                "count"
            ] += 1

            separator = (
                transition[
                    "separator"
                ]
            )

            transition_stats[
                "separators"
            ][separator] += 1

    return patterns


# ============================================================
# ФИНАЛЬНАЯ СЕРИАЛИЗАЦИЯ
# ============================================================

def serialize_patterns(
    patterns
):

    result = []

    for pattern, data in patterns.items():

        # ----------------------------------------------------
        # Fields
        # ----------------------------------------------------

        fields = {}

        for label, stats in (
            data[
                "field_statistics"
            ].items()
        ):

            lengths = stats[
                "lengths"
            ]

            if lengths:

                avg_length = (
                    sum(lengths)
                    / len(lengths)
                )

                min_length = min(
                    lengths
                )

                max_length = max(
                    lengths
                )

            else:

                avg_length = 0
                min_length = 0
                max_length = 0

            fields[label] = {
                "count":
                    stats["count"],

                "avg_length":
                    round(
                        avg_length,
                        2
                    ),

                "min_length":
                    min_length,

                "max_length":
                    max_length,

                "regex_match_rate":
                    round(
                        stats[
                            "regex_matches"
                        ]
                        / stats["count"],
                        4
                    )
                    if stats["count"]
                    else 0,

                "regex_full_match_rate":
                    round(
                        stats[
                            "regex_full_matches"
                        ]
                        / stats["count"],
                        4
                    )
                    if stats["count"]
                    else 0,

                "examples":
                    stats["examples"]
            }

        # ----------------------------------------------------
        # Transitions
        # ----------------------------------------------------

        transitions = []

        for (
            (from_label, to_label),
            transition_data
        ) in data[
            "transitions"
        ].items():

            separator_counter = (
                transition_data[
                    "separators"
                ]
            )

            separators = []

            for separator, count in (
                separator_counter
                .most_common()
            ):

                separators.append(
                    {
                        "text": separator,
                        "count": count,
                        "probability": round(
                            count
                            / transition_data[
                                "count"
                            ],
                            4
                        )
                    }
                )

            transitions.append(
                {
                    "from":
                        from_label,

                    "to":
                        to_label,

                    "count":
                        transition_data[
                            "count"
                        ],

                    "separators":
                        separators
                }
            )

        # ----------------------------------------------------
        # Reference lengths
        # ----------------------------------------------------

        lengths = data[
            "reference_lengths"
        ]

        result.append(
            {
                "pattern":
                    list(pattern),

                "frequency":
                    data["frequency"],

                "reference_length":
                    {
                        "min": min(lengths),
                        "max": max(lengths),
                        "avg": round(
                            sum(lengths)
                            / len(lengths),
                            2
                        )
                    },

                "fields":
                    fields,

                "transitions":
                    transitions
            }
        )

    # Самые частые сначала
    result.sort(
        key=lambda x:
            x["frequency"],
        reverse=True
    )

    return result


# ============================================================
# ОБЩАЯ СТАТИСТИКА
# ============================================================

def print_statistics(
    documents,
    references,
    patterns
):

    print()
    print(
        "=" * 80
    )

    print(
        "ОБЩАЯ СТАТИСТИКА"
    )

    print(
        "=" * 80
    )

    print(
        f"Документов: {documents}"
    )

    print(
        f"Обработано ссылок: "
        f"{len(references)}"
    )

    total_fields = sum(
        len(item["fields"])
        for item in references
    )

    print(
        f"Всего дочерних сущностей: "
        f"{total_fields}"
    )

    if references:

        print(
            "Среднее число полей на ссылку: "
            f"{total_fields / len(references):.2f}"
        )

    print()
    print(
        "=" * 80
    )

    print(
        "СТРУКТУРНЫЕ ПАТТЕРНЫ"
    )

    print(
        "=" * 80
    )

    print(
        f"Всего ссылок: "
        f"{len(references)}"
    )

    print(
        f"Уникальных паттернов: "
        f"{len(patterns)}"
    )

    for i, pattern in enumerate(
        patterns[:30],
        start=1
    ):

        structure = " → ".join(
            pattern["pattern"]
        )

        frequency = pattern[
            "frequency"
        ]

        percentage = (
            frequency
            / len(references)
            * 100
        )

        print(
            f"{i:3d}. "
            f"{frequency:5d} "
            f"({percentage:6.2f}%) "
            f"{structure}"
        )


# ============================================================
# ПЕЧАТЬ ПЕРЕХОДОВ
# ============================================================

def print_transition_statistics(
    patterns
):

    print()
    print(
        "=" * 80
    )

    print(
        "ПРИМЕРЫ ПЕРЕХОДОВ"
    )

    print(
        "=" * 80
    )

    shown = 0

    for pattern in patterns:

        if shown >= 15:
            break

        if not pattern[
            "transitions"
        ]:
            continue

        print()

        print(
            "PATTERN:",
            " → ".join(
                pattern["pattern"]
            )
        )

        print(
            "Частота:",
            pattern["frequency"]
        )

        for transition in pattern[
            "transitions"
        ]:

            print(
                f"  "
                f"{transition['from']}"
                f" → "
                f"{transition['to']}"
            )

            for separator in (
                transition[
                    "separators"
                ][:5]
            ):

                text = separator[
                    "text"
                ]

                if not text:
                    text = "<EMPTY>"

                print(
                    f"      "
                    f"{repr(text):20} "
                    f"{separator['count']:3d} "
                    f"("
                    f"{separator['probability']:.2%}"
                    f")"
                )

        shown += 1


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data",
        help=(
            "Каталог с JSON-разметками"
        )
    )

    parser.add_argument(
        "--output",
        default="patterns_v2.json",
        help=(
            "Файл результатов"
        )
    )

    args = parser.parse_args()

    data_dir = Path(
        args.data
    )

    if not data_dir.exists():

        raise FileNotFoundError(
            data_dir
        )

    # --------------------------------------------------------
    # Ищем JSON
    # --------------------------------------------------------

    json_files = sorted(
        data_dir.glob(
            "*.json"
        )
    )

    print(
        f"Найдено JSON: "
        f"{len(json_files)}"
    )

    all_references = []

    documents_processed = 0

    # --------------------------------------------------------
    # Обработка документов
    # --------------------------------------------------------

    for json_path in json_files:

        try:

            document = load_json(
                json_path
            )

        except Exception as e:

            print(
                f"[ERROR] "
                f"{json_path.name}: "
                f"{e}"
            )

            continue

        documents_processed += 1

        annotations = document.get(
            "annotations",
            []
        )

        # ----------------------------------------------------
        # Ссылки
        # ----------------------------------------------------

        for reference in annotations:

            if (
                reference.get(
                    "label"
                )
                != REFERENCE_LABEL
            ):
                continue

            children = get_children(
                reference
            )

            if not children:
                continue

            analyzed = analyze_reference(
                reference,
                children
            )

            analyzed[
                "document"
            ] = json_path.name

            all_references.append(
                analyzed
            )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    aggregated = aggregate_patterns(
        all_references
    )

    serialized = serialize_patterns(
        aggregated
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print_statistics(
        documents_processed,
        all_references,
        serialized
    )

    print_transition_statistics(
        serialized
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = {
        "statistics": {
            "documents":
                documents_processed,

            "references":
                len(all_references),

            "unique_patterns":
                len(serialized)
        },

        "patterns":
            serialized
    }

    with open(
        args.output,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=4
        )

    print()
    print(
        f"Результат сохранён: "
        f"{args.output}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()