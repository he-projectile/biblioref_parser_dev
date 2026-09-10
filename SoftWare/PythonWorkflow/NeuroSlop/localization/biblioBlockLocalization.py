import argparse
import json
from pathlib import Path

import numpy as np


# ============================================================
# Parameters
# ============================================================

FILTER_SIZE = 3

CWT_MIN_SCALE = 1
CWT_MAX_SCALE = 75

# Коэффициент вычитания среднего перед CWT.
# Соответствует старой реализации plotRecognition.py.
CWT_MEAN_MULTIPLIER = 3

LENGTH_SIGMA = 100
LENGTH_OFFSET = 250


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
# Nonlinear median filter
# ============================================================

def nonlinear_median_filter(signal, window_size):
    """
    Нелинейный медианный фильтр.

    Реализация соответствует старому
    plotRecognition.py.
    """

    signal = np.asarray(
        signal,
        dtype=float
    )

    window_size = int(
        round(window_size)
    )

    if window_size < 3:
        return signal.copy()

    if window_size % 2 == 0:
        window_size += 1

    radius = window_size // 2

    padded = np.pad(
        signal,
        radius,
        mode="edge"
    )

    filtered = np.empty_like(
        signal
    )

    for i in range(len(signal)):

        window = padded[
            i:i + window_size
        ]

        filtered[i] = np.median(
            window
        )

    return filtered


def create_mexican_hat_kernel(width):
    """
    Дискретное Mexican-Hat-подобное ядро.

    width — длина положительной центральной части.

    Для чётного width:
        [-1 ... -1] [1 ... 1] [-1 ... -1]

    Для нечётного width:
        то же ядро, после чего из него
        вычитается среднее значение.
    """

    width = int(width)

    if width < 1:
        raise ValueError("width должен быть >= 1")

    side = width // 2

    # Центральная положительная часть
    positive = np.ones(width, dtype=float)

    # Отрицательные боковые части
    negative = -np.ones(side, dtype=float)

    kernel = np.concatenate([
        negative,
        positive,
        negative
    ])

    # Для нечётной длины центральной части
    # компенсируем DC-составляющую.
    if width % 2 == 1:
        kernel = kernel - np.mean(kernel)

    return kernel

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

    Используется та же реализация, что была
    в исходном plotRecognition.py.
    """

    signal = np.asarray(
        signal,
        dtype=float
    )

    if len(signal) == 0:
        return (
            np.empty((0, 0)),
            np.array([], dtype=int)
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
            np.array([], dtype=int)
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

    for i, width in enumerate(widths):

        # Прямоугольное окно
        kernel = np.ones(
            width,
            dtype=float
        )

#        kernel = create_mexican_hat_kernel(width)

        kernel_length = len(kernel)

        pad_left = kernel_length // 2
        pad_right = kernel_length - 1 - pad_left


        padded_signal = np.pad(
            signal,
            (
                pad_left,
                pad_right
            ),
            mode="constant",
            constant_values=0
        )

        padded_signal = (
            padded_signal
            - np.mean(padded_signal)
            * CWT_MEAN_MULTIPLIER
        )

        values = np.convolve(
            padded_signal,
            kernel,
            mode="valid"
        )

        result[i] = values

    return (
        result,
        widths
    )


# ============================================================
# Detection
# ============================================================

def detect_biblio_block(
    cwt,
    widths
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
        int(end)
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
        / LENGTH_SIGMA ** 2
    )

    scores = scores * length_penalty   

    # --------------------------------------------------------
    # Median filter
    # --------------------------------------------------------

    filtered_scores = nonlinear_median_filter(
        scores,
        FILTER_SIZE
    )

    # --------------------------------------------------------
    # CWT
    # --------------------------------------------------------

    cwt, widths = calculate_cwt(
        filtered_scores,
        CWT_MIN_SCALE,
        CWT_MAX_SCALE
    )

    # --------------------------------------------------------
    # Detection
    # --------------------------------------------------------

    start, end = detect_biblio_block(
        cwt,
        widths
    )

    return {
        "start": start,
        "end": end,
        "scores": scores,
        "filtered_scores": filtered_scores,
        "cwt": cwt,
        "widths": widths,
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

