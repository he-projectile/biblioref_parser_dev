import re
from dataclasses import dataclass
from typing import Optional


# ============================================================
# Результат
# ============================================================

@dataclass
class BibliographyBlock:
    start_line: int
    end_line: int
    score: float

    @property
    def start(self):
        return self.start_line

    @property
    def end(self):
        return self.end_line


# ============================================================
# Нормализация
# ============================================================

def normalize_line(line: str) -> str:
    """
    Минимальная нормализация исключительно для анализа.
    Исходный текст НЕ изменяется.
    """
    line = line.replace("\u00a0", " ")
    line = re.sub(r"\s+", " ", line)
    return line.strip()


# ============================================================
# Признаки
# ============================================================

BIBLIOGRAPHY_HEADERS = [
    r"литература",
    r"список литературы",
    r"список использованной литературы",
    r"использованная литература",
    r"references",
    r"reference list",
    r"bibliography",
    r"literature cited",
    r"works cited",
]

BIBLIOGRAPHY_HEADER_RE = re.compile(
    r"^\s*(?:" + "|".join(BIBLIOGRAPHY_HEADERS) + r")\s*[:.]?\s*$",
    re.IGNORECASE
)


# 1. Иванов...
# 1) Иванов...
# [1] Иванов...
# 1 Иванов...
NUMBERED_REFERENCE_RE = re.compile(
    r"^\s*(?:"
    r"\[\s*\d+\s*\]"
    r"|\d+\s*[\.\)]"
    r"|\d+\s+"
    r")"
)


# [1, 2]
# [1-5]
# [12]
CITATION_RE = re.compile(
    r"\[\s*\d+(?:\s*[-,;]\s*\d+)*\s*\]"
)


DOI_RE = re.compile(
    r"\bdoi\s*:\s*10\.\d{4,9}/\S+",
    re.IGNORECASE
)

DOI_URL_RE = re.compile(
    r"https?://(?:doi\.org/)?10\.\d{4,9}/\S+",
    re.IGNORECASE
)

URL_RE = re.compile(
    r"https?://\S+",
    re.IGNORECASE
)


# Автороподобные конструкции:
#
# Иванов И.И.
# Иванов И. И.
# Smith J.
# Smith J. A.
AUTHOR_RE = re.compile(
    r"\b"
    r"[А-ЯЁA-Z][а-яёa-z-]+"
    r"\s+"
    r"[А-ЯЁA-Z]"
    r"(?:\s*\.\s*[А-ЯЁA-Z])?"
    r"\s*\.",
    re.UNICODE
)


YEAR_RE = re.compile(
    r"\b(?:17|18|19|20|21)\d{2}\b"
)


# ============================================================
# Feature extraction
# ============================================================

def line_features(line: str) -> dict:
    text = normalize_line(line)

    features = {}

    features["empty"] = len(text) == 0

    features["header"] = bool(
        BIBLIOGRAPHY_HEADER_RE.match(text)
    )

    features["numbered"] = bool(
        NUMBERED_REFERENCE_RE.match(text)
    )

    features["citation"] = bool(
        CITATION_RE.search(text)
    )

    features["doi"] = bool(
        DOI_RE.search(text) or DOI_URL_RE.search(text)
    )

    features["url"] = bool(
        URL_RE.search(text)
    )

    features["author"] = bool(
        AUTHOR_RE.search(text)
    )

    features["year"] = bool(
        YEAR_RE.search(text)
    )

    # Небольшой набор дополнительных статистических признаков
    features["length"] = len(text)

    if text:
        features["digit_ratio"] = sum(c.isdigit() for c in text) / len(text)
    else:
        features["digit_ratio"] = 0.0

    features["dots"] = text.count(".")
    features["commas"] = text.count(",")
    features["semicolons"] = text.count(";")
    features["slashes"] = text.count("/")

    return features


# ============================================================
# Reference score
# ============================================================

def reference_score(line: str) -> float:
    """
    Насколько строка похожа на библиографическую запись.
    """

    f = line_features(line)

    if f["empty"]:
        return 0.0

    score = 0.0

    if f["numbered"]:
        score += 4.0

    if f["author"]:
        score += 2.0

    if f["year"]:
        score += 2.0

    if f["doi"]:
        score += 3.0

    if f["url"]:
        score += 1.5

    if f["citation"]:
        score += 1.0

    # Библиографические записи обычно достаточно длинные
    if f["length"] >= 40:
        score += 1.0

    if f["length"] >= 100:
        score += 0.5

    # Много пунктуации характерно для библиографии
    if f["dots"] >= 2:
        score += 0.5

    if f["commas"] >= 1:
        score += 0.5

    return score


# ============================================================
# Header score
# ============================================================

def header_score(line: str) -> float:
    text = normalize_line(line)

    if BIBLIOGRAPHY_HEADER_RE.match(text):
        return 10.0

    return 0.0


# ============================================================
# Проверка блока после кандидата
# ============================================================

def block_support(
    lines,
    start: int,
    window: int = 20
) -> float:
    """
    Проверяем, действительно ли после start идёт
    последовательность библиографических записей.
    """

    scores = []

    end = min(len(lines), start + window)

    for i in range(start, end):
        scores.append(reference_score(lines[i]))

    if not scores:
        return 0.0

    # Сколько строк достаточно сильно похожи на библиографию
    strong = sum(score >= 4.0 for score in scores)

    medium = sum(score >= 2.0 for score in scores)

    return strong * 2.0 + medium * 0.5


# ============================================================
# Поиск кандидатов
# ============================================================

def find_candidates(lines):
    candidates = []

    for i, line in enumerate(lines):

        hscore = header_score(line)

        if hscore > 0:

            support = block_support(
                lines,
                i + 1
            )

            score = hscore + support

            candidates.append(
                (i, score, "header")
            )

    return candidates


# ============================================================
# Поиск конца библиографии
# ============================================================

def find_block_end(
    lines,
    start: int,
    min_reference_score: float = 2.0,
    max_gap: int = 2
) -> int:

    last_good = start
    gap = 0

    for i in range(start, len(lines)):

        line = lines[i]

        score = reference_score(line)

        if score >= min_reference_score:

            last_good = i
            gap = 0

        elif normalize_line(line) == "":

            gap += 1

            if gap > max_gap:
                break

        else:

            gap += 1

            # Если после библиографии пошёл
            # большой непрерывный кусок обычного текста,
            # считаем библиографию законченной.
            if gap > max_gap:
                break

    return last_good


# ============================================================
# Основной алгоритм
# ============================================================

def locate_bibliography(text: str) -> Optional[BibliographyBlock]:

    lines = text.splitlines()

    if not lines:
        return None

    candidates = find_candidates(lines)

    if not candidates:
        return locate_without_header(lines)

    # Выбираем самого сильного кандидата
    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    start, score, _ = candidates[0]

    end = find_block_end(
        lines,
        start + 1
    )

    return BibliographyBlock(
        start_line=start,
        end_line=end,
        score=score
    )


# ============================================================
# Библиография без заголовка
# ============================================================

def locate_without_header(lines):

    best_start = None
    best_score = 0.0

    # Ищем окна, в которых много библиографических строк
    window_size = 10

    for i in range(len(lines)):

        scores = []

        for j in range(
            i,
            min(len(lines), i + window_size)
        ):
            scores.append(
                reference_score(lines[j])
            )

        if not scores:
            continue

        strong = sum(
            score >= 4.0
            for score in scores
        )

        medium = sum(
            score >= 2.0
            for score in scores
        )

        score = (
            strong * 2.0
            + medium * 0.5
        )

        # Библиография обычно расположена
        # ближе к концу документа.
        relative_position = i / max(len(lines), 1)

        if relative_position > 0.5:
            score += 1.0

        if relative_position > 0.7:
            score += 1.0

        if score > best_score:
            best_score = score
            best_start = i

    if best_start is None:
        return None

    end = find_block_end(
        lines,
        best_start
    )

    return BibliographyBlock(
        start_line=best_start,
        end_line=end,
        score=best_score
    )


# ============================================================
# Удобный интерфейс
# ============================================================

def extract_bibliography(text: str):

    block = locate_bibliography(text)

    if block is None:
        return None

    lines = text.splitlines()

    return "\n".join(
        lines[
            block.start_line:
            block.end_line + 1
        ]
    )


# ============================================================
# Тест
# ============================================================


import sys
from pathlib import Path
from tkinter import Tk, filedialog


def get_input_file() -> Path:
    # 1. Файл передан аргументом командной строки
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])

        if not path.exists():
            raise FileNotFoundError(
                f"Файл не найден: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Это не файл: {path}"
            )

        return path

    # 2. Аргумент не передан — открываем проводник
    root = Tk()
    root.withdraw()

    filename = filedialog.askopenfilename(
        title="Выберите текстовый файл",
        filetypes=[
            ("Text files", "*.txt"),
            ("All files", "*.*"),
        ]
    )

    root.destroy()

    if not filename:
        raise SystemExit("Файл не выбран.")

    return Path(filename)


if __name__ == "__main__":

    file_path = get_input_file()

    # Читаем исходный текст
    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:
        text = f.read()

    print(f"Файл: {file_path}")
    print(f"Размер: {len(text)} символов")
    print()

    # ========================================================
    # Здесь вызываем твой существующий локализатор
    # ========================================================

    result = locate_bibliography(text)

    if result is None:
        print("Библиография не найдена.")
    else:
        print(
            f"Найдена библиография:"
        )
        print(
            f"строки {result.start_line + 1}"
            f" - {result.end_line + 1}"
        )
        print(
            f"score = {result.score:.2f}"
        )

        print()
        print("========== БИБЛИОГРАФИЯ ==========")

        lines = text.splitlines()

        print(
            "\n".join(
                lines[
                    result.start_line:
                    result.end_line + 1
                ]
            )
        )