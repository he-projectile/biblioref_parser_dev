import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


REFERENCE_LABEL = "БИБЛ. ССЫЛКА"


def load_machine_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    weights = np.array(
        [pattern["weight"] for pattern in data["patterns"]],
        dtype=float
    )

    line_numbers = np.array(
        [line["line"] for line in data["lines"]],
        dtype=int
    )

    counts = np.array(
        [line["counts"] for line in data["lines"]],
        dtype=float
    )

    return line_numbers, counts, weights


def calculate_score(counts, weights):
    """
    SCORE_i = sum_j count_ij * weight_j
    """
    return counts @ weights


def get_reference_annotations(annotations):
    """
    Рекурсивно ищет все аннотации БИБЛ. ССЫЛКА.
    """

    references = []

    def walk(items):
        for annotation in items:
            if annotation.get("label") == REFERENCE_LABEL:
                references.append(annotation)

            children = annotation.get("children", [])

            if children:
                walk(children)

    walk(annotations)

    return references


def char_to_line(text, char_position):
    """
    Перевод позиции символа в номер строки.

    Нумерация строк начинается с 1.
    """

    return text.count("\n", 0, char_position) + 1


def get_reference_line_bounds(txt_filename, json_filename):
    """
    Возвращает крайние строки библиографического раздела.

    ВАЖНО:
    если между ссылками есть пропуски, они всё равно входят
    в единый эталонный интервал.

    Например:
        ссылки: 100-110
                112-120

    результат:
        [100, 120]
    """

    with open(txt_filename, "r", encoding="utf-8") as f:
        text = f.read()

    with open(json_filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    references = get_reference_annotations(
        data.get("annotations", [])
    )

    if not references:
        return None

    first_lines = []
    last_lines = []

    for reference in references:

        start = reference["start"]
        end = reference["end"]

        start_line = char_to_line(text, start)
        end_line = char_to_line(text, end)

        first_lines.append(start_line)
        last_lines.append(end_line)

    return min(first_lines), max(last_lines)


def low_pass_filter(signal, characteristic_period):
    """
    Butterworth ФНЧ.

    characteristic_period задаётся в строках.
    """

    if characteristic_period <= 1:
        return signal.copy()

    cutoff = 1.0 / characteristic_period

    cutoff = min(max(cutoff, 0.001), 0.99)

    b, a = butter(
        N=2,
        Wn=cutoff,
        btype="low"
    )

    if len(signal) < 10:
        return signal.copy()

    return filtfilt(b, a, signal)


def hysteresis_detection(signal, threshold, hysteresis):
    """
    Гистерезисный детектор.

    Включение:
        signal >= threshold

    Выключение:
        signal < threshold - hysteresis
    """

    upper = threshold
    lower = threshold - hysteresis

    state = np.zeros(len(signal), dtype=bool)

    active = False

    for i, value in enumerate(signal):

        if not active:
            if value >= upper:
                active = True
        else:
            if value < lower:
                active = False

        state[i] = active

    return state


def get_detected_bounds(line_numbers, state):
    """
    Возвращает крайние строки всего обнаруженного участка.

    Если детектор дал несколько отдельных кусков,
    они объединяются в один интервал.
    """

    indices = np.where(state)[0]

    if len(indices) == 0:
        return None

    return (
        int(line_numbers[indices[0]]),
        int(line_numbers[indices[-1]])
    )


def calculate_iou(predicted_bounds, reference_bounds):
    """
    IoU двух цельных интервалов строк.
    """

    if predicted_bounds is None or reference_bounds is None:
        return 0.0

    pred_start, pred_end = predicted_bounds
    ref_start, ref_end = reference_bounds

    intersection_start = max(pred_start, ref_start)
    intersection_end = min(pred_end, ref_end)

    if intersection_start > intersection_end:
        intersection = 0
    else:
        intersection = (
            intersection_end - intersection_start + 1
        )

    union_start = min(pred_start, ref_start)
    union_end = max(pred_end, ref_end)

    union = union_end - union_start + 1

    return intersection / union


def make_plot(
    filename,
    line_numbers,
    score,
    filtered_score,
    state,
    threshold,
    hysteresis,
    characteristic_period,
    reference_bounds
):
    fig, ax = plt.subplots(figsize=(18, 8))

    # =========================================================
    # SCORE
    # =========================================================

    ax.plot(
        line_numbers,
        score,
        linewidth=0.7,
        alpha=0.30,
        label="Исходный SCORE"
    )

    # =========================================================
    # Отфильтрованный SCORE
    # =========================================================

    ax.plot(
        line_numbers,
        filtered_score,
        linewidth=2.0,
        label="SCORE после ФНЧ"
    )

    # =========================================================
    # Порог
    # =========================================================

    ax.axhline(
        threshold,
        linestyle="--",
        linewidth=1.5,
        label=f"Порог = {threshold:g}"
    )

    # =========================================================
    # Нижняя граница гистерезиса
    # =========================================================

    lower_threshold = threshold - hysteresis

    ax.axhline(
        lower_threshold,
        linestyle=":",
        linewidth=1.5,
        label=f"Нижний порог = {lower_threshold:g}"
    )

    # =========================================================
    # Эталонный библиографический раздел
    # =========================================================

    if reference_bounds is not None:

        ref_start, ref_end = reference_bounds

        ax.axvspan(
            ref_start,
            ref_end,
            facecolor="C0",
            alpha=0.05,
            edgecolor="black",
            hatch="///",
            linewidth=0.0,
            label=f"Эталон: строки {ref_start}–{ref_end}"
        )

    # =========================================================
    # Обнаруженный участок
    # =========================================================

    detected_bounds = get_detected_bounds(
        line_numbers,
        state
    )

    if detected_bounds is not None:

        det_start, det_end = detected_bounds

        ax.axvspan(
            det_start,
            det_end,
            facecolor="C0",
            alpha=0.05,
            edgecolor="black",
            hatch="\\\\\\",
            linewidth=0.0,
            label=f"Распознано: строки {det_start}–{det_end}"
        )

    # =========================================================
    # IoU
    # =========================================================

    iou = calculate_iou(
        detected_bounds,
        reference_bounds
    )

    # Выводим IoU прямо на графике
    ax.text(
        0.99,
        0.97,
        f"IoU = {iou:.4f}",
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=14,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.85
        )
    )

    # =========================================================
    # Оформление
    # =========================================================

    ax.set_xlabel("Номер строки")
    ax.set_ylabel("SCORE")

    ax.set_title(
        "Распознавание библиографического раздела\n"
        f"ФНЧ: {characteristic_period:g} строк | "
        f"порог: {threshold:g} | "
        f"гистерезис: {hysteresis:g} | "
        f"IoU: {iou:.4f}"
    )

    ax.grid(True, alpha=0.25)

    ax.legend(
        loc="upper left"
    )

    fig.tight_layout()

    output_filename = filename.with_suffix(".png")

    fig.savefig(
        output_filename,
        dpi=150
    )

    plt.close(fig)

    return output_filename, iou, detected_bounds


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Построение SCORE с ФНЧ, гистерезисом "
            "и сравнением с разметкой."
        )
    )

    parser.add_argument(
        "machine_file",
        type=str,
        help="MACHINE_*.json"
    )

    parser.add_argument(
        "annotation_file",
        type=str,
        help="Размеченный document.json"
    )

    parser.add_argument(
        "--filter",
        type=float,
        default=10.0,
        help=(
            "Характерный период ФНЧ в строках "
            "(по умолчанию 10)"
        )
    )

    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Порог распознавания"
    )

    parser.add_argument(
        "--hysteresis",
        type=float,
        default=1.0,
        help=(
            "Ширина зоны нечувствительности "
            "(по умолчанию 1)"
        )
    )

    args = parser.parse_args()

    machine_filename = Path(args.machine_file)
    annotation_filename = Path(args.annotation_file)

    if not machine_filename.exists():
        print(
            f"Ошибка: MACHINE-файл не найден:\n"
            f"{machine_filename}"
        )
        return

    if not annotation_filename.exists():
        print(
            f"Ошибка: файл разметки не найден:\n"
            f"{annotation_filename}"
        )
        return

    # =========================================================
    # Проверяем TXT
    # =========================================================

    # MACHINE_document.json
    #
    # annotation:
    # document.json
    #
    # txt:
    # document.txt

    txt_filename = annotation_filename.with_suffix(".txt")

    if not txt_filename.exists():
        print(
            f"Ошибка: TXT-файл не найден:\n"
            f"{txt_filename}"
        )
        return

    # =========================================================
    # Загружаем MACHINE
    # =========================================================

    line_numbers, counts, weights = load_machine_file(
        machine_filename
    )

    # =========================================================
    # SCORE
    # =========================================================

    score = calculate_score(
        counts,
        weights
    )

    # =========================================================
    # ФНЧ
    # =========================================================

    filtered_score = low_pass_filter(
        score,
        args.filter
    )

    # =========================================================
    # Гистерезис
    # =========================================================

    state = hysteresis_detection(
        filtered_score,
        args.threshold,
        args.hysteresis
    )

    # =========================================================
    # Эталонные границы
    # =========================================================

    reference_bounds = get_reference_line_bounds(
        txt_filename,
        annotation_filename
    )

    if reference_bounds is None:
        print(
            "Предупреждение: БИБЛ. ССЫЛКА "
            "в разметке не найдена."
        )

    # =========================================================
    # График
    # =========================================================

    output_filename, iou, detected_bounds = make_plot(
        machine_filename,
        line_numbers,
        score,
        filtered_score,
        state,
        args.threshold,
        args.hysteresis,
        args.filter,
        reference_bounds
    )

    # =========================================================
    # Результат
    # =========================================================

    print()
    print("========================================")
    print("РЕЗУЛЬТАТ")
    print("========================================")

    print(f"Входной файл : {machine_filename}")
    print(f"Разметка     : {annotation_filename}")
    print(f"TXT          : {txt_filename}")

    print()
    print(f"ФНЧ          : {args.filter:g} строк")
    print(f"Порог        : {args.threshold:g}")
    print(f"Гистерезис   : {args.hysteresis:g}")

    print()

    if reference_bounds is not None:
        print(
            f"Эталон       : "
            f"{reference_bounds[0]}–{reference_bounds[1]}"
        )
    else:
        print("Эталон       : нет")

    if detected_bounds is not None:
        print(
            f"Распознано   : "
            f"{detected_bounds[0]}–{detected_bounds[1]}"
        )
    else:
        print("Распознано   : нет")

    print()
    print(f"IoU          : {iou:.6f}")

    print()
    print(f"График       : {output_filename}")
    print("========================================")


if __name__ == "__main__":
    main()