import argparse
import json
import re
from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# CONFIG
# ============================================================

BIB_LABEL = "БИБЛ. ССЫЛКА"

# Минимальная длина найденной ссылки
MIN_REFERENCE_LENGTH = 20

# Максимальная длина кандидата.
# Нужна защита от случайного захвата огромного текста.
MAX_REFERENCE_LENGTH = 2000

# Минимальная уверенность паттерна.
MIN_PATTERN_FREQUENCY = 2


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# LOAD TXT
# ============================================================

def load_text(
    annotation,
    json_path
):

    document_name = annotation.get(
        "document"
    )

    if not document_name:

        raise ValueError(
            f"{json_path}: "
            f"нет поля document"
        )

    txt_path = (
        json_path.parent
        / document_name
    )

    if not txt_path.exists():

        raise FileNotFoundError(
            f"Не найден TXT:\n"
            f"{txt_path}"
        )

    with open(
        txt_path,
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    text = text.replace(
        "\r",
        "\n"
    )

    # Перенос строки рассматриваем как пробел
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def normalize_for_match(text):

    text = normalize_text(
        text
    )

    text = text.lower()

    return text


# ============================================================
# ANNOTATIONS
# ============================================================

def get_references(
    annotation
):

    references = []

    for item in annotation.get(
        "annotations",
        []
    ):

        if item.get(
            "label"
        ) != BIB_LABEL:

            continue

        references.append(
            {
                "start": int(
                    item["start"]
                ),

                "end": int(
                    item["end"]
                ),

                "text": item.get(
                    "text",
                    ""
                )
            }
        )

    references.sort(
        key=lambda x: x["start"]
    )

    return references


# ============================================================
# FIELD REGEX
# ============================================================

FIELD_REGEX = {

    "ГОД ПУБЛИКАЦИИ":
        re.compile(
            r"(?<!\d)(?:19|20)\d{2}(?!\d)"
        ),

    "СТРАНИЦЫ":
        re.compile(
            r"(?<!\d)"
            r"\d{1,5}"
            r"\s*[-–—]"
            r"\s*"
            r"\d{1,5}"
            r"(?!\d)"
        ),

    "DOI":
        re.compile(
            r"10\.\d{4,9}/"
            r"[^\s\"<>]+",
            re.IGNORECASE
        ),

    "URL":
        re.compile(
            r"https?://[^\s\"<>]+",
            re.IGNORECASE
        ),

    "ISBN":
        re.compile(
            r"\bISBN(?:-|\s)*"
            r"(?:97[89][- ]?)?"
            r"[\dXx][- \dXx]{8,20}"
        ),

    "ISSN":
        re.compile(
            r"\b\d{4}[-–]\d{3}[\dXx]\b"
        ),
}


# ============================================================
# FIELD FIND
# ============================================================

def find_field(
    text,
    label,
    start=0
):

    regex = FIELD_REGEX.get(
        label
    )

    if regex is None:

        return None

    match = regex.search(
        text,
        pos=start
    )

    if match is None:

        return None

    return {
        "start": match.start(),

        "end": match.end(),

        "text": match.group()
    }


# ============================================================
# PATTERN INFORMATION
# ============================================================

def get_pattern_fields(
    pattern
):

    return [
        field
        for field in pattern
        if field in FIELD_REGEX
    ]


# ============================================================
# BUILD ANCHOR REGEX
# ============================================================

def build_anchor_regex(
    pattern
):
    """
    Выбирает наиболее информативные поля,
    по которым можно искать кандидата.

    Например:

        AUTHORS
        TITLE
        JOURNAL
        YEAR
        PAGES

    превращается в поиск:

        YEAR ... PAGES

    """

    fields = get_pattern_fields(
        pattern
    )

    if not fields:

        return None

    # Приоритет якорей
    priority = [
        "DOI",
        "URL",
        "ГОД ПУБЛИКАЦИИ",
        "СТРАНИЦЫ",
        "ISBN",
        "ISSN",
    ]

    for field in priority:

        if field in fields:

            return field

    return fields[0]


# ============================================================
# EXTRACT ANCHOR POSITIONS
# ============================================================

def find_anchors(
    text,
    field
):

    regex = FIELD_REGEX.get(
        field
    )

    if regex is None:

        return []

    result = []

    for match in regex.finditer(
        text
    ):

        result.append(
            {
                "start": match.start(),

                "end": match.end(),

                "text": match.group()
            }
        )

    return result


# ============================================================
# PREFIX / SUFFIX
# ============================================================

def clean_reference_boundary(
    text,
    start,
    end
):

    # --------------------------------------------------------
    # LEFT
    # --------------------------------------------------------

    while start < end:

        char = text[
            start
        ]

        if char.isspace():

            start += 1

        else:

            break

    # --------------------------------------------------------
    # RIGHT
    # --------------------------------------------------------

    while end > start:

        char = text[
            end - 1
        ]

        if char.isspace():

            end -= 1

        else:

            break

    return (
        start,
        end
    )


# ============================================================
# FIND REFERENCE START
# ============================================================

def find_reference_start(
    text,
    anchor_start,
    previous_anchor=None
):
    """
    Ищем начало библиографической записи
    перед якорем.

    На первом этапе используем:
    - номер списка;
    - начало строки;
    - предыдущую пустую строку.
    """

    search_start = 0

    if previous_anchor is not None:

        search_start = previous_anchor

    region = text[
        search_start:
        anchor_start
    ]

    # --------------------------------------------------------
    # Последний перенос строки
    # --------------------------------------------------------

    line_start = region.rfind(
        "\n"
    )

    if line_start >= 0:

        candidate = (
            search_start
            + line_start
            + 1
        )

    else:

        candidate = search_start

    # --------------------------------------------------------
    # Проверяем номер библиографической записи
    #
    # 1.
    # 2)
    # [3]
    # 15.
    # --------------------------------------------------------

    number_match = re.search(
        r"(?:^|\n)"
        r"\s*"
        r"(?:"
        r"\[\s*\d+\s*\]"
        r"|"
        r"\d+\s*[\.)]"
        r")"
        r"\s*",
        text[
            search_start:
            anchor_start
        ]
    )

    if number_match:

        candidate = (
            search_start
            + number_match.start()
        )

        # Если совпадение начинается
        # с \n — убираем его
        if (
            text[candidate:candidate + 1]
            == "\n"
        ):

            candidate += 1

    return candidate


# ============================================================
# FIND REFERENCE END
# ============================================================

def find_reference_end(
    text,
    anchor_end
):
    """
    После якоря ищем естественную границу.

    Приоритет:
    1. конец строки;
    2. следующая нумерованная запись;
    3. следующий пустой блок.
    """

    limit = min(
        len(text),
        anchor_end
        + MAX_REFERENCE_LENGTH
    )

    region = text[
        anchor_end:
        limit
    ]

    # --------------------------------------------------------
    # Следующая нумерованная запись
    # --------------------------------------------------------

    next_number = re.search(
        r"\n\s*"
        r"(?:"
        r"\[\s*\d+\s*\]"
        r"|"
        r"\d+\s*[\.)]"
        r")"
        r"\s+",
        region
    )

    # --------------------------------------------------------
    # Если нашли следующую запись
    # --------------------------------------------------------

    if next_number:

        return (
            anchor_end
            + next_number.start()
        )

    # --------------------------------------------------------
    # Иначе конец строки
    # --------------------------------------------------------

    newline = region.find(
        "\n"
    )

    if newline >= 0:

        return (
            anchor_end
            + newline
        )

    return limit


# ============================================================
# BUILD CANDIDATE
# ============================================================

def build_candidate(
    text,
    anchor,
    previous_anchor
):

    start = find_reference_start(
        text,
        anchor["start"],
        previous_anchor
    )

    end = find_reference_end(
        text,
        anchor["end"]
    )

    start, end = (
        clean_reference_boundary(
            text,
            start,
            end
        )
    )

    if end <= start:

        return None

    if (
        end - start
        < MIN_REFERENCE_LENGTH
    ):

        return None

    if (
        end - start
        > MAX_REFERENCE_LENGTH
    ):

        return None

    return {
        "start": start,

        "end": end,

        "text": text[
            start:end
        ]
    }


# ============================================================
# MATCH PATTERN
# ============================================================

def match_pattern(
    text,
    pattern
):
    """
    Ищет кандидатов по конкретному
    структурному паттерну.
    """

    anchor_field = build_anchor_regex(
        pattern
    )

    if anchor_field is None:

        return []

    anchors = find_anchors(
        text,
        anchor_field
    )

    candidates = []

    previous_anchor = None

    for anchor in anchors:

        candidate = build_candidate(
            text,
            anchor,
            previous_anchor
        )

        if candidate is None:

            continue

        candidate[
            "pattern"
        ] = list(pattern)

        candidate[
            "anchor"
        ] = anchor_field

        candidates.append(
            candidate
        )

        previous_anchor = (
            anchor["end"]
        )

    return candidates


# ============================================================
# DEDUPLICATE CANDIDATES
# ============================================================

def deduplicate_candidates(
    candidates
):

    if not candidates:

        return []

    candidates = sorted(
        candidates,
        key=lambda x: (
            x["start"],
            x["end"]
        )
    )

    result = []

    for candidate in candidates:

        duplicate = False

        for existing in result:

            # Полное совпадение
            if (
                candidate["start"]
                == existing["start"]

                and

                candidate["end"]
                == existing["end"]
            ):

                duplicate = True

                break

            # Один кандидат содержит другой
            if (
                candidate["start"]
                >= existing["start"]

                and

                candidate["end"]
                <= existing["end"]
            ):

                duplicate = True

                break

        if not duplicate:

            result.append(
                candidate
            )

    return result


# ============================================================
# OVERLAP
# ============================================================

def overlap(
    a_start,
    a_end,
    b_start,
    b_end
):

    return (
        a_start < b_end
        and
        b_start < a_end
    )


# ============================================================
# EXACT MATCH
# ============================================================

def exact_match(
    candidate,
    reference
):

    return (
        candidate["start"]
        == reference["start"]

        and

        candidate["end"]
        == reference["end"]
    )


# ============================================================
# IoU
# ============================================================

def span_iou(
    candidate,
    reference
):

    intersection_start = max(
        candidate["start"],
        reference["start"]
    )

    intersection_end = min(
        candidate["end"],
        reference["end"]
    )

    intersection = max(
        0,
        intersection_end
        - intersection_start
    )

    candidate_length = (
        candidate["end"]
        - candidate["start"]
    )

    reference_length = (
        reference["end"]
        - reference["start"]
    )

    union = (
        candidate_length
        + reference_length
        - intersection
    )

    if union <= 0:

        return 0.0

    return (
        intersection
        / union
    )


# ============================================================
# EVALUATE DOCUMENT
# ============================================================

def evaluate_document(
    candidates,
    references
):
    """
    Один candidate может быть сопоставлен
    максимум с одной gold reference.

    Сначала exact match.
    Затем overlap.
    """

    used_references = set()

    true_positive = 0

    exact_positive = 0

    partial_positive = 0

    matches = []

    false_positive = []

    # --------------------------------------------------------
    # Candidates
    # --------------------------------------------------------

    for candidate in candidates:

        best_reference = None

        best_iou = 0.0

        best_index = None

        # ----------------------------------------------
        # Ищем лучшую gold reference
        # ----------------------------------------------

        for index, reference in enumerate(
            references
        ):

            if index in used_references:

                continue

            iou = span_iou(
                candidate,
                reference
            )

            if iou > best_iou:

                best_iou = iou

                best_reference = (
                    reference
                )

                best_index = index

        # ----------------------------------------------
        # Exact
        # ----------------------------------------------

        if (
            best_reference is not None

            and

            exact_match(
                candidate,
                best_reference
            )
        ):

            true_positive += 1

            exact_positive += 1

            used_references.add(
                best_index
            )

            matches.append(
                {
                    "candidate": candidate,

                    "reference": (
                        best_reference
                    ),

                    "iou": best_iou,

                    "type": "exact"
                }
            )

        # ----------------------------------------------
        # Partial overlap
        # ----------------------------------------------

        elif (
            best_reference is not None

            and

            best_iou >= 0.5
        ):

            true_positive += 1

            partial_positive += 1

            used_references.add(
                best_index
            )

            matches.append(
                {
                    "candidate": candidate,

                    "reference": (
                        best_reference
                    ),

                    "iou": best_iou,

                    "type": "partial"
                }
            )

        else:

            false_positive.append(
                candidate
            )

    false_negative = (
        len(references)
        - len(used_references)
    )

    return {
        "tp": true_positive,

        "fp": len(
            false_positive
        ),

        "fn": false_negative,

        "exact": exact_positive,

        "partial": partial_positive,

        "matches": matches,

        "false_positive": (
            false_positive
        )
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    tp,
    fp,
    fn
):

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2
        * precision
        * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return (
        precision,
        recall,
        f1
    )


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(
    data_dir,
    patterns_path
):

    data_dir = Path(
        data_dir
    )

    pattern_data = load_json(
        patterns_path
    )

    # --------------------------------------------------------
    # Получаем паттерны
    # --------------------------------------------------------

    patterns = []

    for item in pattern_data.get(
        "patterns",
        []
    ):

        frequency = item.get(
            "frequency",
            0
        )

        if (
            frequency
            < MIN_PATTERN_FREQUENCY
        ):

            continue

        pattern = tuple(
            item["pattern"]
        )

        patterns.append(
            {
                "pattern": pattern,

                "frequency": frequency
            }
        )

    print()
    print(
        "=" * 80
    )

    print(
        "ЗАГРУЗКА ПАТТЕРНОВ"
    )

    print(
        "=" * 80
    )

    print(
        f"Паттернов: "
        f"{len(patterns)}"
    )

    # --------------------------------------------------------
    # JSON files
    # --------------------------------------------------------

    json_files = sorted(
        data_dir.glob(
            "*.json"
        )
    )

    print(
        f"Документов: "
        f"{len(json_files)}"
    )

    # --------------------------------------------------------
    # Global statistics
    # --------------------------------------------------------

    total_tp = 0

    total_fp = 0

    total_fn = 0

    total_exact = 0

    total_partial = 0

    all_false_positive = []

    document_statistics = []

    # --------------------------------------------------------
    # Documents
    # --------------------------------------------------------

    for json_path in json_files:

        try:

            annotation = load_json(
                json_path
            )

            text = load_text(
                annotation,
                json_path
            )

        except Exception as e:

            print(
                f"[ERROR] "
                f"{json_path.name}: "
                f"{e}"
            )

            continue

        references = get_references(
            annotation
        )

        all_candidates = []

        # ----------------------------------------------------
        # Pattern matching
        # ----------------------------------------------------

        for pattern_info in patterns:

            pattern = pattern_info[
                "pattern"
            ]

            candidates = match_pattern(
                text,
                pattern
            )

            all_candidates.extend(
                candidates
            )

        # ----------------------------------------------------
        # Dedup
        # ----------------------------------------------------

        all_candidates = (
            deduplicate_candidates(
                all_candidates
            )
        )

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        statistics = evaluate_document(
            all_candidates,
            references
        )

        total_tp += statistics[
            "tp"
        ]

        total_fp += statistics[
            "fp"
        ]

        total_fn += statistics[
            "fn"
        ]

        total_exact += statistics[
            "exact"
        ]

        total_partial += statistics[
            "partial"
        ]

        all_false_positive.extend(
            statistics[
                "false_positive"
            ]
        )

        document_statistics.append(
            {
                "document": (
                    json_path.name
                ),

                "references": len(
                    references
                ),

                "candidates": len(
                    all_candidates
                ),

                "tp": statistics[
                    "tp"
                ],

                "fp": statistics[
                    "fp"
                ],

                "fn": statistics[
                    "fn"
                ],

                "exact": statistics[
                    "exact"
                ],

                "partial": statistics[
                    "partial"
                ],
            }
        )

        print(
            f"{json_path.name:60} "
            f"gold={len(references):3d} "
            f"pred={len(all_candidates):3d} "
            f"TP={statistics['tp']:3d} "
            f"FP={statistics['fp']:3d} "
            f"FN={statistics['fn']:3d}"
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    precision, recall, f1 = (
        calculate_metrics(
            total_tp,
            total_fp,
            total_fn
        )
    )

    exact_recall = (
        total_exact
        / (
            total_exact
            + total_partial
            + total_fn
        )
        if (
            total_exact
            + total_partial
            + total_fn
        ) > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "РЕЗУЛЬТАТ"
    )

    print(
        "=" * 80
    )

    print(
        f"TP: {total_tp}"
    )

    print(
        f"FP: {total_fp}"
    )

    print(
        f"FN: {total_fn}"
    )

    print()

    print(
        f"Precision: "
        f"{precision:.4f}"
    )

    print(
        f"Recall:    "
        f"{recall:.4f}"
    )

    print(
        f"F1:        "
        f"{f1:.4f}"
    )

    print()

    print(
        f"Exact boundary: "
        f"{total_exact}"
    )

    print(
        f"Partial overlap: "
        f"{total_partial}"
    )

    print(
        f"Exact recall: "
        f"{exact_recall:.4f}"
    )

    # --------------------------------------------------------
    # False positives
    # --------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "ПРИМЕРЫ FALSE POSITIVE"
    )

    print(
        "=" * 80
    )

    for candidate in (
        all_false_positive[:20]
    ):

        print()

        print(
            f"{candidate['start']}:"
            f"{candidate['end']}"
        )

        print(
            repr(
                candidate["text"]
            )
        )

        print(
            "Pattern:",
            " → ".join(
                candidate["pattern"]
            )
        )

        print(
            "Anchor:",
            candidate["anchor"]
        )

    return {
        "tp": total_tp,

        "fp": total_fp,

        "fn": total_fn,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "exact": total_exact,

        "partial": total_partial,

        "exact_recall": exact_recall,

        "documents": (
            document_statistics
        )
    }


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(
    output_path,
    result
):

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=4
        )

    print()

    print(
        f"Результаты сохранены: "
        f"{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Поиск библиографических ссылок "
            "по индуцированным структурным "
            "паттернам."
        )
    )

    parser.add_argument(
        "data",
        help=(
            "Каталог с TXT + JSON"
        )
    )

    parser.add_argument(
        "--patterns",
        default="patterns.json",
        help=(
            "patterns.json"
        )
    )

    parser.add_argument(
        "--output",
        default="pattern_results.json",
        help=(
            "Файл результатов"
        )
    )

    args = parser.parse_args()

    result = process_dataset(
        args.data,
        args.patterns
    )

    save_result(
        args.output,
        result
    )


if __name__ == "__main__":
    main()