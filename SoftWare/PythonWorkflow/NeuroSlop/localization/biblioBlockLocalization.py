import json
from pathlib import Path

import numpy as np


# ============================================================
# PARAMETERS
# ============================================================

# Нелинейный медианный фильтр
FILTER_SIZE = 3

# Диапазон ширины окна CWT
CWT_MIN_SCALE = 1
CWT_MAX_SCALE = 75

# Нормализация score по длине строки
LENGTH_SIGMA = 100
LENGTH_OFFSET = 250

# Множитель вычитания среднего значения в CWT
CWT_MEAN_MULTIPLIER = 2.5


# ============================================================
# DATA LOADING
# ============================================================

def load_machine_file(filename):
    """
    Загрузка MACHINE_*.json.
    """

    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def load_weights(filename):
    """
    Загрузка весов паттернов из patterns.json.

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

    Возвращает numpy-массив весов.
    """

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    patterns = data.get("patterns", [])

    weights = []

    for pattern in patterns:
        weights.append(
            float(pattern.get("weight", 0.0))
        )

    return np.asarray(
        weights,
        dtype=float
    )


# ============================================================
# SCORE
# ============================================================

def calculate_scores(
    machine_data,
    weights
):
    """
    Рассчитать SCORE для всех строк.

    SCORE сначала вычисляется как:

        score = counts @ weights

    Затем применяется нормализация по длине строки:

        score *= exp(
            -(length - LENGTH_OFFSET)^2
            / LENGTH_SIGMA^2
        )

    Важно:
        веса НЕ хранятся в MACHINE JSON.
    """

    lines = machine_data["lines"]

    if not lines:
        return np.array([], dtype=float)

    counts = np.asarray(
        [
            line["counts"]
            for line in lines
        ],
        dtype=float
    )

    lengths = np.asarray(
        [
            line["length"]
            for line in lines
        ],
        dtype=float
    )

    weights = np.asarray(
        weights,
        dtype=float
    )

    if counts.shape[1] != len(weights):
        raise ValueError(
            "Number of weights does not match "
            "number of pattern counts: "
            f"{len(weights)} weights vs "
            f"{counts.shape[1]} patterns"
        )

    # --------------------------------------------------------
    # Linear combination of pattern activations
    # --------------------------------------------------------

    scores = counts @ weights

    # --------------------------------------------------------
    # Length normalization
    # --------------------------------------------------------

    lengths = np.maximum(
        lengths,
        1
    )

    length_factor = np.exp(
        -(
            (lengths - LENGTH_OFFSET) ** 2
        )
        / LENGTH_SIGMA ** 2
    )

    scores *= length_factor

    return scores


# ============================================================
# NONLINEAR MEDIAN FILTER
# ============================================================

def nonlinear_median_filter(
    signal,
    window_size=FILTER_SIZE
):
    """
    Нелинейный медианный фильтр.

    Логика полностью соответствует текущему
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


# ============================================================
# CWT
# ============================================================

def calculate_cwt(
    signal,
    min_width=CWT_MIN_SCALE,
    max_width=CWT_MAX_SCALE
):
    """
    CWT-подобное преобразование прямоугольным окном.

    Это намеренно оставляет ту же реализацию,
    которая использовалась в plotRecognition.py.

    Для каждой ширины окна:

        1. дополняем сигнал нулями;
        2. вычитаем среднее * CWT_MEAN_MULTIPLIER;
        3. сворачиваем с прямоугольным окном;
        4. получаем строку CWT.
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

    min_width = int(
        round(min_width)
    )

    max_width = int(
        round(max_width)
    )

    if min_width < 1:
        min_width = 1

    if max_width < min_width:
        max_width = min_width

    max_width = min(
        max_width,
        len(signal)
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

        kernel = np.ones(
            width,
            dtype=float
        )

        pad_left = width // 2

        pad_right = (
            width
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

    return result, widths


# ============================================================
# DETECTION
# ============================================================

def detect_biblio_block(
    scores,
    filter_size=FILTER_SIZE,
    cwt_min_scale=CWT_MIN_SCALE,
    cwt_max_scale=CWT_MAX_SCALE
):
    """
    Найти границы библиографического блока.

    Pipeline:

        SCORE
          ↓
        median filter
          ↓
        CWT
          ↓
        global argmax
          ↓
        block bounds

    Возвращает:

        detected_bounds
        filtered_scores
        cwt
        widths
    """

    scores = np.asarray(
        scores,
        dtype=float
    )

    if len(scores) == 0:
        return (
            None,
            np.array([], dtype=float),
            np.empty((0, 0)),
            np.array([], dtype=int)
        )

    # --------------------------------------------------------
    # Median filter
    # --------------------------------------------------------

    filtered_scores = nonlinear_median_filter(
        scores,
        filter_size
    )

    # --------------------------------------------------------
    # CWT
    # --------------------------------------------------------

    cwt, widths = calculate_cwt(
        filtered_scores,
        cwt_min_scale,
        cwt_max_scale
    )

    if cwt.size == 0:
        return (
            None,
            filtered_scores,
            cwt,
            widths
        )

    # --------------------------------------------------------
    # Global maximum
    # --------------------------------------------------------

    best_index = np.unravel_index(
        np.argmax(cwt),
        cwt.shape
    )

    best_width = widths[
        best_index[0]
    ]

    best_center = best_index[1]

    # --------------------------------------------------------
    # Convert center + width into line bounds
    # --------------------------------------------------------

    det_start = max(
        1,
        best_center
        - best_width // 2
        + 1
    )

    det_end = min(
        len(filtered_scores),
        det_start
        + best_width
        - 1
    )

    detected_bounds = [
        int(det_start),
        int(det_end)
    ]

    return (
        detected_bounds,
        filtered_scores,
        cwt,
        widths
    )


# ============================================================
# MAIN LOCALIZATION FUNCTION
# ============================================================

def localizeBiblioBlock(
    machine_filename,
    patterns_filename,
    filter_size=FILTER_SIZE,
    cwt_min_scale=CWT_MIN_SCALE,
    cwt_max_scale=CWT_MAX_SCALE
):
    """
    Полный pipeline локализации библиографического блока.

    Вход:

        MACHINE_*.json
        patterns.json

    Выход:

        {
            "start": ...,
            "end": ...,
            "scores": ...,
            "filtered_scores": ...,
            "cwt": ...,
            "widths": ...
        }

    Эта функция является единственной точкой,
    через которую другие модули должны запускать
    локализацию.

    Таким образом:

        plot
        IoU calculation
        optimizer

    используют абсолютно одинаковый алгоритм.
    """

    machine_filename = Path(
        machine_filename
    )

    patterns_filename = Path(
        patterns_filename
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    machine_data = load_machine_file(
        machine_filename
    )

    weights = load_weights(
        patterns_filename
    )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    scores = calculate_scores(
        machine_data,
        weights
    )

    # --------------------------------------------------------
    # Detection
    # --------------------------------------------------------

    (
        detected_bounds,
        filtered_scores,
        cwt,
        widths
    ) = detect_biblio_block(
        scores,
        filter_size,
        cwt_min_scale,
        cwt_max_scale
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    if detected_bounds is None:
        start = None
        end = None
    else:
        start, end = detected_bounds

    return {
        "start": start,
        "end": end,
        "scores": scores,
        "filtered_scores": filtered_scores,
        "cwt": cwt,
        "widths": widths,
    }


# ============================================================
# IoU
# ============================================================

def calculate_iou(
    reference_bounds,
    detected_bounds
):
    """
    Intersection over Union для двух интервалов строк.

    Интервалы считаются inclusive:

        [start, end]

    Это соответствует текущей реализации
    calculateOveralIou.py.
    """

    if (
        reference_bounds is None
        or detected_bounds is None
    ):
        return 0.0

    ref_start, ref_end = reference_bounds
    det_start, det_end = detected_bounds

    intersection_start = max(
        ref_start,
        det_start
    )

    intersection_end = min(
        ref_end,
        det_end
    )

    if intersection_end < intersection_start:
        intersection = 0
    else:
        intersection = (
            intersection_end
            - intersection_start
            + 1
        )

    reference_length = (
        ref_end
        - ref_start
        + 1
    )

    detected_length = (
        det_end
        - det_start
        + 1
    )

    union = (
        reference_length
        + detected_length
        - intersection
    )

    if union == 0:
        return 0.0

    return (
        intersection / union
    )


# ============================================================
# ANNOTATIONS
# ============================================================

REFERENCE_LABEL = "БИБЛ. ССЫЛКА"


def get_reference_annotations(
    annotation_data
):
    """
    Рекурсивно найти все annotations
    с label == БИБЛ. ССЫЛКА.

    Возвращает:

        [
            (start, end),
            ...
        ]
    """

    result = []

    def recursive_search(obj):

        if isinstance(obj, dict):

            if (
                obj.get("label")
                == REFERENCE_LABEL
            ):

                if (
                    "start" in obj
                    and "end" in obj
                ):

                    result.append(
                        (
                            int(obj["start"]),
                            int(obj["end"])
                        )
                    )

            for value in obj.values():
                recursive_search(value)

        elif isinstance(obj, list):

            for item in obj:
                recursive_search(item)

    recursive_search(
        annotation_data
    )

    return result


def char_to_line(
    text,
    char_pos
):
    """
    Перевести позицию символа в номер строки.

    Нумерация строк начинается с 1.
    """

    before = text[:char_pos]

    return (
        before.count("\n")
        + 1
    )


def get_reference_line_bounds(
    text,
    annotations
):
    """
    Перевести char annotations библиографических
    ссылок в общий диапазон строк.

    Если имеется несколько ссылок:

        [10, 20]
        [25, 40]
        [100, 120]

    результат:

        (строка_10, строка_120)
    """

    if not annotations:
        return None

    starts = []
    ends = []

    for start, end in annotations:

        start_line = char_to_line(
            text,
            start + 1
        )

        end_line = char_to_line(
            text,
            end + 1
        )

        starts.append(
            start_line
        )

        ends.append(
            end_line
        )

    return (
        min(starts),
        max(ends)
    )


def load_reference_bounds(
    annotation_filename,
    text_filename=None
):
    """
    Загрузить annotation JSON и соответствующий TXT,
    после чего вернуть эталонные границы блока.

    Если text_filename не указан, TXT берётся
    из annotation_filename с заменой расширения.
    """

    annotation_filename = Path(
        annotation_filename
    )

    if text_filename is None:
        text_filename = (
            annotation_filename.with_suffix(
                ".txt"
            )
        )
    else:
        text_filename = Path(
            text_filename
        )

    with open(
        annotation_filename,
        "r",
        encoding="utf-8"
    ) as f:

        annotation_data = json.load(f)

    with open(
        text_filename,
        "r",
        encoding="utf-8"
    ) as f:

        text = f.read()

    annotations = get_reference_annotations(
        annotation_data
    )

    return get_reference_line_bounds(
        text,
        annotations
    )


# ============================================================
# LOCALIZATION + IoU
# ============================================================

def localizeBiblioBlockWithIoU(
    machine_filename,
    patterns_filename,
    annotation_filename,
    text_filename=None,
    filter_size=FILTER_SIZE,
    cwt_min_scale=CWT_MIN_SCALE,
    cwt_max_scale=CWT_MAX_SCALE
):
    """
    Локализовать библиографический блок
    и сразу рассчитать IoU с разметкой.

    Удобно для:

        plot
        evaluation
        optimizer
    """

    result = localizeBiblioBlock(
        machine_filename=machine_filename,
        patterns_filename=patterns_filename,
        filter_size=filter_size,
        cwt_min_scale=cwt_min_scale,
        cwt_max_scale=cwt_max_scale
    )

    detected_bounds = None

    if (
        result["start"] is not None
        and result["end"] is not None
    ):
        detected_bounds = [
            result["start"],
            result["end"]
        ]

    reference_bounds = load_reference_bounds(
        annotation_filename,
        text_filename
    )

    iou = calculate_iou(
        reference_bounds,
        detected_bounds
    )

    result["reference_bounds"] = (
        reference_bounds
    )

    result["detected_bounds"] = (
        detected_bounds
    )

    result["iou"] = iou

    return result