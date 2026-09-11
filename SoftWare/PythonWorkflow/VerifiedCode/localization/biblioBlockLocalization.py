import argparse
import json
from pathlib import Path

import numpy as np


# ============================================================
# Parameters
# ============================================================


CWT_MIN_SCALE = 1
CWT_MAX_SCALE = 75

CWT_MEAN_MULTIPLIER = 2

LENGTH_SIGMA = 100
LENGTH_OFFSET = 222

SEARCH_START_K = 0.908
SEARCH_START_K_ALTERNATIVE = SEARCH_START_K**2
SEARCH_START_B = -2.23
SEARCH_START_STD = 11.45

# ============================================================
# Loading
# ============================================================

def load_machine_file(filename):
    """
    Загружает MACHINE_*.json.

    Ожидаемый формат:

    {
        "patterns": [...],
        "lines": [
            {
                "line": 1,
                "length": 83,
                "counts": [0, 1, 0, ...],
                "text": "..."
            },
            ...
        ]
    }
    """

    filename = Path(filename)

    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def loadLocalizationData(machine_filename):
    machine_data = load_machine_file(machine_filename)

    counts = np.asarray(
        [line["counts"] for line in machine_data["lines"]],
        dtype=float
    )

    lengths = np.asarray(
        [line["length"] for line in machine_data["lines"]],
        dtype=float
    )

    return {
        "counts": counts,
        "lengths": lengths,
    }

def load_localization_data(machine_filename):
    """
    Загружает MACHINE JSON один раз.

    Возвращает данные, необходимые для локализации.
    """

    machine_data = load_machine_file(machine_filename)

    counts = np.asarray(
        [line["counts"] for line in machine_data["lines"]],
        dtype=float
    )

    lengths = np.asarray(
        [line["length"] for line in machine_data["lines"]],
        dtype=float
    )

    return {
        "counts": counts,
        "lengths": lengths,
    }

def load_patterns(filename):
    """
    Загружает patterns.json.

    Ожидаемый формат:

    {
        "patterns": [
            {
                "name": "...",
                "regex": "...",
                "weight": 1.0
            },
            ...
        ]
    }

    Возвращает:
        patterns
        weights
    """

    filename = Path(filename)

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "patterns" not in data:
        raise ValueError(
            f"В файле {filename} отсутствует ключ 'patterns'"
        )

    patterns = data["patterns"]

    weights = []

    for index, pattern in enumerate(patterns):

        if "weight" not in pattern:
            raise ValueError(
                f"У паттерна №{index} "
                f"({pattern.get('name', '<без имени>')}) "
                f"отсутствует поле 'weight'"
            )

        try:
            weight = float(pattern["weight"])
        except (TypeError, ValueError):
            raise ValueError(
                f"Некорректный weight у паттерна №{index}: "
                f"{pattern['weight']}"
            )

        weights.append(weight)

    return (
        patterns,
        np.asarray(weights, dtype=float)
    )


# ============================================================
# Score calculation
# ============================================================

def calculate_scores(machine_data, weights):
    """
    Вычисляет SCORE для каждой строки.

    Для каждой строки:

        SCORE = sum(count[i] * weight[i])

    где:
        count[i]  — количество срабатываний i-го паттерна
        weight[i] — вес i-го паттерна
    """

    lines = machine_data.get("lines", [])

    if not lines:
        return np.array([], dtype=float)

    scores = []

    number_of_patterns = len(weights)

    for line in lines:

        if "counts" not in line:
            raise ValueError(
                f"В строке {line.get('line', '?')} "
                f"отсутствует поле 'counts'"
            )

        counts = np.asarray(
            line["counts"],
            dtype=float
        )

        if len(counts) != number_of_patterns:
            raise ValueError(
                "Количество элементов 'counts' "
                "не совпадает с количеством паттернов.\n"
                f"Строка: {line.get('line', '?')}\n"
                f"counts: {len(counts)}\n"
                f"patterns: {number_of_patterns}"
            )

        score = np.dot(
            counts,
            weights
        )

        scores.append(score)

    return np.asarray(
        scores,
        dtype=float
    )

# ============================================================
# CALCULATE START
# ============================================================

def calculate_search_start(
    number_of_lines
):
    """
    Вычисляет начало области поиска в строках.

    position = k * document_lines + b

    Возвращает индекс строки в формате
    Python (0-based).
    """

    search_start_line_gain = (
        SEARCH_START_K_ALTERNATIVE
        * number_of_lines
    )

    search_start_line_offsets = (
        SEARCH_START_K
        * number_of_lines
        + SEARCH_START_B
        - SEARCH_START_STD*3
    )

    search_start_line = max(
        0,
        min(
            search_start_line_gain,
            search_start_line_offsets,
            number_of_lines - 1
        )
    )

    return int(search_start_line)

# ============================================================
# CWT
# ============================================================

def calculate_cwt(
    signal,
    min_width,
    max_width
):
    """
    Вычисляет CWT прямоугольным окном.

    Возвращает:

        cwt
            numpy.ndarray размера:
            [количество масштабов, количество строк]

        widths
            numpy.ndarray с реальными ширинами окон.

        mean_threshold
            значение:

                np.mean(padded_signal)
                * CWT_MEAN_MULTIPLIER

            для каждого масштаба.
    """

    signal = np.asarray(
        signal,
        dtype=float
    )

    if len(signal) == 0:
        return (
            np.empty((0, 0)),
            np.array([], dtype=int),
            np.array([], dtype=float)
        )

    min_width = int(min_width)
    max_width = int(max_width)

    if min_width < 1:
        min_width = 1

    max_width = min(
        max_width,
        len(signal)
    )

    if min_width > max_width:
        return (
            np.empty((0, len(signal))),
            np.array([], dtype=int),
            np.array([], dtype=float)
        )

    widths = np.arange(
        min_width,
        max_width + 1
    )

    result = np.zeros(
        (
            len(widths),
            len(signal)
        ),
        dtype=float
    )

    mean_thresholds = np.zeros(
        len(widths),
        dtype=float
    )

    for i, width in enumerate(widths):

        kernel = np.ones(
            width,
            dtype=float
        )

        kernel_length = len(kernel)

        pad_left = kernel_length // 2
        pad_right = (
            kernel_length
            - 1
            - pad_left
        )

        padded_signal = np.pad(
            signal,
            (
                pad_left,
                pad_right
            ),
            mode="constant",
            constant_values=0
        )

        mean_threshold = (
            np.mean(padded_signal)
            * CWT_MEAN_MULTIPLIER
        )

        mean_thresholds[i] = mean_threshold

        padded_signal = (
            padded_signal
            - mean_threshold
        )

        values = np.convolve(
            padded_signal,
            kernel,
            mode="valid"
        )

        result[i] = values

    return (
        result,
        widths,
        mean_thresholds
    )


# ============================================================
# Detection
# ============================================================

def detect_biblio_block(
    cwt,
    widths,
    mean_thresholds
):
    """
    Находит положение максимума CWT и преобразует его
    в границы библиографического блока.

    Возвращает:

        start
        end

    Нумерация строк начинается с 1.
    """

    if cwt.size == 0:
        return None, None

    if len(widths) == 0:
        return None, None

    # Индекс максимального значения CWT.
    best_index = np.unravel_index(
        np.argmax(cwt),
        cwt.shape
    )

    width_index = best_index[0]
    best_center = best_index[1]

    best_width = widths[
        width_index
    ]

    mean_threshold = mean_thresholds[
        width_index
    ]    

    # Преобразование индекса массива
    # в номер строки.
    start = max(
        1,
        best_center
        - best_width // 2
        + 1
    )

    end = min(
        len(cwt[0]),
        start
        + best_width
        - 1
    )

    return (
        int(start),
        int(end),
        float(mean_threshold)
    )


# ============================================================
# Main localization function
# ============================================================

def localizeBiblioBlockData(
    localization_data,
    weights
):
    counts = localization_data["counts"]
    lengths = localization_data["lengths"]

    weights = np.asarray(
        weights,
        dtype=float
    )

    if counts.ndim != 2:
        raise ValueError(
            "counts должен быть двумерным массивом"
        )

    if counts.shape[1] != len(weights):
        raise ValueError(
            f"Количество весов ({len(weights)}) "
            f"не совпадает с количеством паттернов "
            f"({counts.shape[1]})"
        )

    # --------------------------------------------------------
    # Pattern score
    # --------------------------------------------------------

    scores = np.dot(
        counts,
        weights
    )

    # --------------------------------------------------------
    # Length penalty
    # --------------------------------------------------------

    length_penalty = np.exp(
        -(lengths - LENGTH_OFFSET) ** 2
        / (2*LENGTH_SIGMA ** 2)
    )

    filtered_scores = scores * length_penalty   

    # --------------------------------------------------------
    # Search range
    # --------------------------------------------------------

    document_line_count = len(
        filtered_scores
    )

    search_start_line = calculate_search_start(
        document_line_count
    )

    search_signal = filtered_scores[
        search_start_line:
    ]

    # --------------------------------------------------------
    # CWT
    # --------------------------------------------------------

    search_cwt, widths, mean_thresholds  = calculate_cwt(
        search_signal,
        CWT_MIN_SCALE,
        CWT_MAX_SCALE
    )


    # --------------------------------------------------------
    # Detection
    # --------------------------------------------------------

    relative_start, relative_end, mean_threshold = (
        detect_biblio_block(
            search_cwt,
            widths,
            mean_thresholds
        )
    )


    if relative_start is not None:

        start = (
            relative_start
            + search_start_line
        )

        end = (
            relative_end
            + search_start_line
        )

    else:

        start = None
        end = None

    return {
        "start": start, 
        "end": end,
        "scores": scores,
        "filtered_scores": filtered_scores,
        "cwt": search_cwt,
        "search_start": search_start_line,
        "widths": widths,
        "mean_threshold": mean_threshold,
    }

def localizeBiblioBlock(
    machine_filename,
    patterns_filename,
    weights_override=None
):
    """
    Основная функция локализации библиографического блока.

    Параметры
    ---------
    machine_filename:
        MACHINE_*.json с результатами patternRecognition.py.

    patterns_filename:
        patterns.json, содержащий паттерны и их веса.

    Возвращает
    ----------
    dict:

        {
            "start": int | None,
            "end": int | None,

            "scores": numpy.ndarray,

            "filtered_scores": numpy.ndarray,

            "cwt": numpy.ndarray,

            "widths": numpy.ndarray
        }

    ВАЖНО:
        Функция НЕ читает annotation-файл
        и НЕ вычисляет IoU.

        Она отвечает только за локализацию.
    """

    # --------------------------------------------------------
    # Load input
    # --------------------------------------------------------

    machine_data = load_machine_file(
        machine_filename
    )

    localization_data = loadLocalizationData(
        machine_filename
    )

    patterns, weights = load_patterns(
        patterns_filename
    )

    if weights_override is not None:
        weights = np.asarray(
            weights_override,
            dtype=float
        )

    pattern_count = len(
        machine_data["lines"][0]["counts"]
    )
    if len(weights) != pattern_count:
        raise ValueError(
            f"Количество весов ({len(weights)}) "
            f"не совпадает с количеством паттернов "
            f"({pattern_count})"
        )        

    # --------------------------------------------------------
    # Return everything needed by other modules
    # --------------------------------------------------------

    return localizeBiblioBlockData(
        localization_data,
        weights
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Локализация блока библиографических ссылок"
        )
    )

    parser.add_argument(
        "machine_file",
        type=Path,
        help="MACHINE_*.json"
    )

    parser.add_argument(
        "--patterns",
        required=True,
        type=Path,
        help="patterns.json"
    )

    args = parser.parse_args()

    result = localizeBiblioBlock(
        args.machine_file,
        args.patterns
    )

    print(
        f"Распознано: "
        f"строки {result['start']}–{result['end']}"
    )

    print(
        f"Количество строк: "
        f"{len(result['scores'])}"
    )

    print(
        f"Размер CWT: "
        f"{result['cwt'].shape}"
    )


if __name__ == "__main__":
    main()

