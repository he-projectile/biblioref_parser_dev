#!/usr/bin/env python3

import re
import json
import argparse

from pathlib import Path
from dataclasses import dataclass


# ============================================================
# Micro-pattern
# ============================================================

@dataclass
class Pattern:
    name: str
    regex: str
    description: str


PATTERNS = [

    Pattern(
        "YEAR",
        r"(?<!\d)(17|18|19|20|21)\d{2}(?!\d)",
        "Четырёхзначный год"
    ),

    Pattern(
        "SURNAME_INITIALS",
        r"\b[А-ЯЁ][а-яё]{2,20}\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.",
        "Фамилия И.О."
    ),

    Pattern(
        "INITIALS_SURNAME",
        r"\b[А-ЯЁ]\.\s*[А-ЯЁ]\.\s*[А-ЯЁ][а-яё]{2,20}",
        "И.О. Фамилия"
    ),

    Pattern(
        "SURNAME_INITIAL",
        r"\b[А-ЯЁ][а-яё]{2,20}\s+[А-ЯЁ]\.",
        "Фамилия И."
    ),

    Pattern(
        "INITIAL_SURNAME",
        r"\b[А-ЯЁ]\.\s*[А-ЯЁ][а-яё]{2,20}",
        "И. Фамилия"
    ),

    Pattern(
        "DOI",
        r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b",
        "DOI"
    ),

    Pattern(
        "URL",
        r"(?:https?://|www\.)[^\s]+",
        "URL"
    ),

    Pattern(
        "ISBN",
        r"\bISBN(?:-1[03])?\s*[:\-]?\s*[\dXx][\dXx\- ]{8,20}\b",
        "ISBN"
    ),

    Pattern(
        "ISSN",
        r"\bISSN\s*[:\-]?\s*\d{4}[-–]\d{3}[\dXx]\b",
        "ISSN"
    ),

    Pattern(
        "DOUBLE_SLASH",
        r"//",
        "Разделитель //"
    ),

    Pattern(
        "COLON",
        r":",
        "Двоеточие"
    ),

    Pattern(
        "DASH",
        r"(?:^|\s)[\-–—]\s*",
        "Тире-разделитель"
    ),

    Pattern(
        "DOT_DASH",
        r"\.\s*[\-–—]",
        "Конструкция .- / .– / .—"
    ),

    Pattern(
        "SEMICOLON",
        r";",
        "Точка с запятой"
    ),

    Pattern(
        "PAGES",
        r"\b(?:С|с|P|p|стр|страниц[аы])\.?\s*\d+(?:\s*[-–—]\s*\d+)?",
        "Страницы"
    ),

    Pattern(
        "VOLUME",
        r"\b(?:Т|т|Vol|vol)\.?\s*\d+",
        "Том"
    ),

    Pattern(
        "NUMBER",
        r"\b(?:№|No\.?|N)\s*\d+",
        "Номер"
    ),

    Pattern(
        "EDITION",
        r"\b(?:Изд|изд|издательство|Издательство)\b",
        "Издательство"
    ),

    Pattern(
        "PLACE_PUBLISHER",
        r"\b(?:М|СПб|Л|Мн|Киев|Москва|Санкт[- ]Петербург)\b",
        "Место издания"
    ),

    Pattern(
        "PARENTHESIS",
        r"\([^()\n]{1,80}\)",
        "Скобочная конструкция"
    ),

    Pattern(
        "LIST_NUMBER",
        r"^\s*(?:\[\d+\]|\d+[\.\)]|\(\d+\))\s+",
        "Нумерация записи"
    ),

    Pattern(
        "QUOTATION",
        r"[«»\"„“”]",
        "Кавычки"
    ),

    Pattern(
        "SLASH",
        r"/",
        "Слэш"
    ),
]


# ============================================================
# Loading
# ============================================================

def load_weights(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Compile
# ============================================================

def compile_patterns():

    return [
        (
            pattern,
            re.compile(
                pattern.regex,
                re.UNICODE
            )
        )
        for pattern in PATTERNS
    ]


# ============================================================
# Split document
# ============================================================

def split_lines(text):

    lines = []

    offset = 0

    for index, raw_line in enumerate(
        text.splitlines(keepends=True)
    ):

        line = raw_line.rstrip("\r\n")

        start = offset
        end = start + len(line)

        lines.append({
            "index": index,
            "start": start,
            "end": end,
            "text": line
        })

        offset += len(raw_line)

    return lines


# ============================================================
# Feature vector
# ============================================================

def get_feature_vector(text, compiled_patterns):

    vector = {}

    for pattern, regex in compiled_patterns:

        matches = regex.findall(text)

        vector[pattern.name] = len(matches)

    return vector


# ============================================================
# Scalar score
# ============================================================

def get_score(vector, weights):

    score = 0.0

    for name, value in vector.items():

        weight = weights.get(name, 0.0)

        score += value * weight

    return score


# ============================================================
# Process
# ============================================================

def process_document(text, weights):

    compiled_patterns = compile_patterns()

    lines = split_lines(text)

    for line in lines:

        vector = get_feature_vector(
            line["text"],
            compiled_patterns
        )

        score = get_score(
            vector,
            weights
        )

        line["features"] = vector
        line["score"] = score

    return lines


# ============================================================
# Save
# ============================================================

def save_result(path, lines, weights):

    result = {
        "weights": weights,
        "patterns": [
            {
                "name": p.name,
                "regex": p.regex,
                "description": p.description
            }
            for p in PATTERNS
        ],
        "lines": lines
    }

    with open(path, "w", encoding="utf-8") as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# Console
# ============================================================

def print_scores(lines):

    print()

    print(
        f"{'LINE':>6} | "
        f"{'SCORE':>8} | "
        f"TEXT"
    )

    print("-" * 120)

    for line in lines:

        text = line["text"].replace(
            "\t",
            " "
        )

        if len(text) > 90:
            text = text[:87] + "..."

        print(
            f"{line['index']:6d} | "
            f"{line['score']:8.3f} | "
            f"{text}"
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Построчное вычисление сигнала библиографических микропаттернов"
    )

    parser.add_argument(
        "input",
        help="Путь к входному TXT-файлу"
    )

    parser.add_argument(
        "--weights",
        "-w",
        required=True,
        help="Путь к JSON-файлу с коэффициентами микропаттернов"
    )

    parser.add_argument(
        "--log",
        "-l",
        required=True,
        help="Куда сохранить лог вычисления признаков и сигнала"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Входной файл не найден: {input_path}"
        )

    # --------------------------------------------------------
    # Weights
    # --------------------------------------------------------

    weights_path = Path(args.weights)

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Файл коэффициентов не найден: {weights_path}"
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Processing
    # --------------------------------------------------------

    text = input_path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    weights = load_weights(
        weights_path
    )

    lines = process_document(
        text,
        weights
    )

    save_result(
        log_path,
        lines,
        weights
    )

    print_scores(lines)

    print()
    print(f"Входной файл:      {input_path}")
    print(f"Файл коэффициентов: {weights_path}")
    print(f"Лог:               {log_path}")


if __name__ == "__main__":
    main()
