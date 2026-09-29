import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import butter, filtfilt
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches

from biblioBlockLocalization import localizeBiblioBlock


REFERENCE_LABEL = "БИБЛ. ССЫЛКА"


# ============================================================
# Butterworth
# ============================================================

BUTTERWORTH_ORDER = 3

# Нормированная частота среза:
# 0 < cutoff < 1
#
# Чем меньше значение, тем сильнее сглаживание.
BUTTERWORTH_CUTOFF = 0.4


# ============================================================
# Гистерезис
# ============================================================

# Пороги задаются относительно максимума filtered_score.
#
# HIGH_THRESHOLD_RATIO:
#   выше этого значения сигнал должен подняться,
#   чтобы начался библиографический блок.
#
# LOW_THRESHOLD_RATIO:
#   ниже этого значения сигнал должен опуститься,
#   чтобы блок закончился.
#
# LOW должен быть меньше HIGH.
HIGH_THRESHOLD_RATIO = 0.55
LOW_THRESHOLD_RATIO = 0.30


# ============================================================
# Работа с разметкой
# ============================================================

def get_reference_annotations(annotation_data):
    result = []

    def recursive_search(obj):

        if isinstance(obj, dict):

            if obj.get("label") == REFERENCE_LABEL:

                if "start" in obj and "end" in obj:
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

    recursive_search(annotation_data)

    return result


def char_to_line(text, char_pos):
    return text[:char_pos].count("\n") + 1


def get_reference_line_bounds(text, annotations):

    if not annotations:
        return None

    starts = []
    ends = []

    for start, end in annotations:

        starts.append(
            char_to_line(
                text,
                start + 1
            )
        )

        ends.append(
            char_to_line(
                text,
                end + 1
            )
        )

    return (
        min(starts),
        max(ends)
    )


def load_reference_bounds(annotation_filename):

    if annotation_filename is None:
        return None

    annotation_filename = Path(
        annotation_filename
    )

    txt_filename = annotation_filename.with_suffix(
        ".txt"
    )

    with open(
        annotation_filename,
        "r",
        encoding="utf-8"
    ) as f:
        annotation_data = json.load(f)

    with open(
        txt_filename,
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


def calculate_iou(
    reference_bounds,
    detected_bounds
):

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

    return intersection / union

# ============================================================
# Butterworth
# ============================================================

def apply_butterworth(
    signal,
    order=BUTTERWORTH_ORDER,
    cutoff=BUTTERWORTH_CUTOFF
):

    signal = np.asarray(
        signal,
        dtype=float
    )

    if len(signal) < 10:
        return signal.copy()

    b, a = butter(
        order,
        cutoff,
        btype="low"
    )

    # filtfilt не имеет фазового сдвига.
    filtered = filtfilt(
        b,
        a,
        signal
    )

    return filtered


# ============================================================
# Hysteresis
# ============================================================

def detect_hysteresis_bounds(
    signal,
    search_start,
    high_threshold,
    low_threshold
):
    """
    Гистерезисная детекция.

    Сначала ищем пересечение HIGH.

    После начала блока продолжаем его, пока сигнал
    не опустится ниже LOW.

    Возвращает:
        start,
        end

    Нумерация строк — 1-based.
    """

    signal = np.asarray(
        signal,
        dtype=float
    )

    n = len(signal)

    if n == 0:
        return None, None

    search_start = max(
        0,
        min(
            int(search_start),
            n - 1
        )
    )

    # --------------------------------------------------------
    # Поиск начала
    # --------------------------------------------------------

    start_index = None

    for i in range(
        search_start,
        n
    ):

        if signal[i] >= high_threshold:

            start_index = i
            break

    if start_index is None:
        return None, None

    # --------------------------------------------------------
    # Поиск конца
    # --------------------------------------------------------

    end_index = n - 1

    for i in range(
        start_index + 1,
        n
    ):

        if signal[i] < low_threshold:

            end_index = i - 1
            break

    # Перевод в нумерацию строк.
    start = start_index + 1
    end = end_index + 1

    return start, end


# ============================================================
# Общий стиль графика
# ============================================================

def make_base_figure():

    fig = plt.figure(
        figsize=(20, 4.8),
        dpi=300
    )

    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[30, 0.5, 5],
        height_ratios=[0.1, 1],
        hspace=0.12,
        wspace=0.08
    )

    fig.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.90,
        bottom=0.1
    )

    ax_score = fig.add_subplot(
        gs[1, 0]
    )

    ax_legend = fig.add_subplot(
        gs[1, 2]
    )

    ax_legend.axis("off")

    return (
        fig,
        ax_score,
        ax_legend
    )


def setup_score_axis(
    ax,
    number_of_lines
):

    ax.set_xlim(
        0.5,
        number_of_lines + 0.5
    )

    ax.xaxis.set_major_locator(
        MaxNLocator(
            integer=True,
            nbins=15
        )
    )

    ax.set_xlabel(
        "Номер строки"
    )

    ax.set_ylabel(
        "SCORE"
    )

    ax.grid(
        True,
        alpha=0.3
    )


# ============================================================
# Разметка на графике
# ============================================================

def add_reference(
    ax,
    reference_bounds
):

    if reference_bounds is None:
        return

    ref_start, ref_end = reference_bounds

    plt.rcParams[
        "hatch.linewidth"
    ] = 0.20

    ax.axvspan(
        ref_start - 0.5,
        ref_end + 0.5,
        facecolor="none",
        edgecolor="black",
        hatch="//",
        linewidth=0.20,
        label=(
            f"Эталон: строки "
            f"{ref_start}–{ref_end}"
        )
    )


def add_search_start(
    ax,
    search_start
):

    # search_start у локализатора 0-based.
    # На графике — 1-based.

    search_start_line = search_start + 1

    ax.axvline(
        search_start_line,
        linestyle="--",
        linewidth=1.5,
        label=(
            f"Начало поиска "
            f"({search_start_line} строка)"
        )
    )


def add_detected_bounds(
    ax,
    detected_bounds
):

    if detected_bounds is None:
        return

    det_start, det_end = detected_bounds

    plt.rcParams[
        "hatch.linewidth"
    ] = 0.20

    ax.axvspan(
        det_start - 0.5,
        det_end + 0.5,
        facecolor="none",
        edgecolor="black",
        hatch="\\\\",
        linewidth=0.20,
        label=(
            f"Распознано: строки "
            f"{det_start}–{det_end}"
        )
    )


# ============================================================
# Основной график
# ============================================================

def make_plot(
    scores,
    filtered_scores,
    butterworth_scores,
    search_start,
    high_threshold,
    low_threshold,
    detected_bounds,
    reference_bounds,
    output_filename,
    source_filename
):

    fig, ax_score, ax_legend = make_base_figure()

    x = np.arange(
        1,
        len(scores) + 1
    )

    setup_score_axis(
        ax_score,
        len(scores)
    )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    ax_score.step(
        x,
        scores,
        linewidth=1.0,
        where="mid",
        label="SCORE"
    )

    # --------------------------------------------------------
    # SCORE после фильтра длины
    # --------------------------------------------------------

    ax_score.step(
        x,
        filtered_scores,
        linewidth=1.5,
        where="mid",
        label="SCORE после фильтра длины"
    )

    # --------------------------------------------------------
    # Butterworth
    # --------------------------------------------------------

    ax_score.plot(
        x,
        butterworth_scores,
        linewidth=2.0,
        label="SCORE после ФНЧ Butterworth"
    )

    # --------------------------------------------------------
    # Порог HIGH
    # --------------------------------------------------------

    ax_score.axhline(
        high_threshold,
        linestyle="--",
        linewidth=1.5,
        color = "red",
        label=(
            f"HIGH threshold = "
            f"{high_threshold:.3f}"
        )
    )

    # --------------------------------------------------------
    # Порог LOW
    # --------------------------------------------------------

    ax_score.axhline(
        low_threshold,
        linestyle="--",
        linewidth=1.5,
        color="maroon",
        label=(
            f"LOW threshold = "
            f"{low_threshold:.3f}"
        )
    )

    # --------------------------------------------------------
    # Начало поиска
    # --------------------------------------------------------

    add_search_start(
        ax_score,
        search_start
    )

    # --------------------------------------------------------
    # Эталон
    # --------------------------------------------------------

    add_reference(
        ax_score,
        reference_bounds
    )

    # --------------------------------------------------------
    # Распознанный блок
    # --------------------------------------------------------

    add_detected_bounds(
        ax_score,
        detected_bounds
    )

    # --------------------------------------------------
    # IoU
    # --------------------------------------------------

    IoUvalue = calculate_iou(
        reference_bounds,
        detected_bounds
    )    

    ax_score.set_title(
        "SCORE, ФНЧ Butterworth и гистерезисная детекция"
    )

    handles, labels = (
        ax_score.get_legend_handles_labels()
    )

    if IoUvalue is not None:
        # Создаем невидимый маркер для строки с текстом
        empty_handle = mpatches.Rectangle((0, 0), 0, 0, fill=False, edgecolor='none', visible=False)
        
        # Добавляем в конец списков (или в начало, если хотите IoU сверху)
        handles.append(empty_handle)
        labels.append(f"IoU = {IoUvalue:.4f}")

    ax_legend.legend(
        handles,
        labels,
        loc="center left"
    )

    plt.suptitle(
        f"Выделение библиографического блока из документа\n"
        f"{source_filename}",
        fontsize=14,
        y=0.98
    )

    plt.savefig(
        output_filename,
        dpi=300
    )

    plt.close(fig)

    


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "machine_file",
        type=Path
    )

    parser.add_argument(
        "--annotation-file",
        type=Path,
        nargs="?",
        default=None
    )

    parser.add_argument(
        "--patterns",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".")
    )

    parser.add_argument(
        "--butter-order",
        type=int,
        default=BUTTERWORTH_ORDER
    )

    parser.add_argument(
        "--butter-cutoff",
        type=float,
        default=BUTTERWORTH_CUTOFF
    )

    parser.add_argument(
        "--high-ratio",
        type=float,
        default=HIGH_THRESHOLD_RATIO
    )

    parser.add_argument(
        "--low-ratio",
        type=float,
        default=LOW_THRESHOLD_RATIO
    )

    args = parser.parse_args()

    if not 0 < args.butter_cutoff < 1:
        raise ValueError(
            "butter-cutoff должен быть в диапазоне (0, 1)"
        )

    if not (
        0 <= args.low_ratio
        < args.high_ratio
    ):
        raise ValueError(
            "Должно выполняться "
            "0 <= low-ratio < high-ratio"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    source_filename = (
        args.machine_file
        .with_suffix(".txt")
        .name
    )

    # --------------------------------------------------------
    # Получаем данные текущего локализатора
    # --------------------------------------------------------

    result = localizeBiblioBlock(
        args.machine_file,
        args.patterns
    )

    scores = np.asarray(
        result["scores"],
        dtype=float
    )

    filtered_scores = np.asarray(
        result["filtered_scores"],
        dtype=float
    )

    search_start = int(
        result["search_start"]
    )

    # --------------------------------------------------------
    # Butterworth
    # --------------------------------------------------------

    butterworth_scores = apply_butterworth(
        filtered_scores,
        order=args.butter_order,
        cutoff=args.butter_cutoff
    )

    # --------------------------------------------------------
    # Пороги
    #
    # Берём максимум Butterworth-сигнала.
    # --------------------------------------------------------

    signal_max = np.max(
        butterworth_scores
    )

    high_threshold = (
        signal_max
        * args.high_ratio
    )

    low_threshold = (
        signal_max
        * args.low_ratio
    )

    # --------------------------------------------------------
    # Гистерезис
    # --------------------------------------------------------

    detected_start, detected_end = (
        detect_hysteresis_bounds(
            butterworth_scores,
            search_start,
            high_threshold,
            low_threshold
        )
    )

    detected_bounds = None

    if (
        detected_start is not None
        and detected_end is not None
    ):
        detected_bounds = (
            detected_start,
            detected_end
        )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    reference_bounds = load_reference_bounds(
        args.annotation_file
    )

    # --------------------------------------------------------
    # График
    # --------------------------------------------------------

    output_filename = (
        args.output_dir
        / (
            f"{args.machine_file.stem}"
            "_butterworth_hysteresis.png"
        )
    )

    make_plot(
        scores=scores,
        filtered_scores=filtered_scores,
        butterworth_scores=butterworth_scores,
        search_start=search_start,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        detected_bounds=detected_bounds,
        reference_bounds=reference_bounds,
        output_filename=output_filename,
        source_filename=source_filename
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print()

    print(
        f"Источник: {source_filename}"
    )

    print(
        f"Начало поиска: "
        f"строка {search_start + 1}"
    )

    print(
        f"Butterworth order: "
        f"{args.butter_order}"
    )

    print(
        f"Butterworth cutoff: "
        f"{args.butter_cutoff}"
    )

    print(
        f"HIGH threshold: "
        f"{high_threshold:.6f}"
    )

    print(
        f"LOW threshold: "
        f"{low_threshold:.6f}"
    )

    print(
        f"Распознано: "
        f"{detected_bounds}"
    )

    print(
        f"Эталон: "
        f"{reference_bounds}"
    )

    print(
        f"Output: "
        f"{output_filename}"
    )


if __name__ == "__main__":
    main()