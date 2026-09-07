import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


REFERENCE_LABEL = "БИБЛ. ССЫЛКА"


def load_machine_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def calculate_score(machine_data):
    patterns = machine_data["patterns"]
    lines = machine_data["lines"]

    weights = np.array(
        [pattern["weight"] for pattern in patterns],
        dtype=float
    )

    scores = []

    for line in lines:
        counts = np.array(
            line["counts"],
            dtype=float
        )

        score = np.dot(counts, weights)
        scores.append(score)

    return np.array(scores, dtype=float)


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
    before = text[:char_pos]

    line = before.count("\n") + 1

    return line


def get_reference_line_bounds(text, annotations):
    if not annotations:
        return None

    starts = []
    ends = []

    for start, end in annotations:
        start_line = char_to_line(text, start+1)
        end_line = char_to_line(text, end+1)

        starts.append(start_line)
        ends.append(end_line)

    return min(starts), max(ends)


def nonlinear_median_filter(signal, window_size):
    window_size = int(round(window_size))

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

    filtered = np.empty_like(signal)

    for i in range(len(signal)):
        window = padded[i:i + window_size]
        filtered[i] = np.median(window)

    return filtered


def calculate_iou(reference_bounds, detected_bounds):
    if reference_bounds is None or detected_bounds is None:
        return 0.0

    ref_start, ref_end = reference_bounds
    det_start, det_end = detected_bounds

    intersection_start = max(ref_start, det_start)
    intersection_end = min(ref_end, det_end)

    if intersection_end < intersection_start:
        intersection = 0
    else:
        intersection = intersection_end - intersection_start + 1

    reference_length = ref_end - ref_start + 1
    detected_length = det_end - det_start + 1

    union = reference_length + detected_length - intersection

    if union == 0:
        return 0.0

    return intersection / union

def calculate_cwt(signal, min_width, max_width):
    signal = np.asarray(signal)

    widths = np.arange(
        min_width,
        min(max_width, len(signal)) + 1
    )

    result = np.zeros(
        (len(widths), len(signal))
    )

    for i, width in enumerate(widths):
        kernel = np.ones(width)

        pad_left = width // 2
        pad_right = width - 1 - pad_left

        padded_signal = np.pad(
            signal,
            (pad_left, pad_right),
            mode="constant",
            constant_values=0
        )

        # Сдвигаем весь дополненный массив вниз на его среднее значение
        padded_signal = padded_signal - np.mean(padded_signal)

        values = np.convolve(
            padded_signal,
            kernel,
            mode="valid"
        )

        result[i] = values

    return result, widths

def make_plot(
    scores,
    filtered_scores,
    cwt,
    scales,
    reference_bounds,
    detected_bounds,
    IoUvalue,
    output_filename
):
    fig = plt.figure(figsize=(20, 12), dpi=300)
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[30, 1, 5],
        height_ratios=[1.5, 1],
        hspace=0.18,
        wspace=0.08
    )    

    fig.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.95,
        bottom=0.05
    )

    ax_cwt = fig.add_subplot(gs[0, 0])
    ax_score = fig.add_subplot(gs[1, 0], sharex=ax_cwt)

    ax_cbar = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, 2])

    ax_legend.axis("off")

    x = np.arange(
        1,
        len(scores) + 1
    )

    # ---------------------------------------------------------
    # CWT HEATMAP
    # ---------------------------------------------------------

    max_abs = np.max(np.abs(cwt))

    if max_abs == 0:
        max_abs = 1.0

    image = ax_cwt.imshow(
        cwt,
        aspect="auto",
        origin="lower",
        extent=[
            1,
            len(scores),
            scales[0],
            scales[-1]
        ],
        cmap="RdBu_r",
        vmin=-max_abs,
        vmax=max_abs
    )

    ax_cwt.set_ylabel("Ширина окна, строк")
    ax_cwt.set_title("CWT прямоугольного окна")

    fig.colorbar(
        image,
        cax=ax_cbar,
        label="Интенсивность CWT"
    )

    # ---------------------------------------------------------
    # SCORE
    # ---------------------------------------------------------

    ax_score.plot(
        x,
        scores,
        linewidth=1.0,
        label="SCORE"
    )

    ax_score.plot(
        x,
        filtered_scores,
        linewidth=2.0,
        label="SCORE после нелинейного ФНЧ"
    )

    # ---------------------------------------------------------
    # REFERENCE / DETECTED AREAS
    # ---------------------------------------------------------

    if reference_bounds is not None:
        ref_start, ref_end = reference_bounds

        for ax in [ax_cwt, ax_score]:
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

    if detected_bounds is not None:
        det_start, det_end = detected_bounds

        for ax in [ax_cwt, ax_score]:
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

    # ---------------------------------------------------------
    # SCORE AXIS
    # ---------------------------------------------------------

    ax_score.set_xlabel("Номер строки")
    ax_score.set_ylabel("SCORE")

    ax_score.set_title(
        "SCORE и результат нелинейного обнаружения"
    )

    ax_score.grid(
        True,
        alpha=0.3
    )

    handles, labels = ax_score.get_legend_handles_labels()

    ax_legend.legend(
        handles,
        labels,
        loc="center left"
    )
    ax_legend.text(
        0,
        0.35,
        f"IoU = {IoUvalue:.4f}",
        transform=ax_legend.transAxes
    )

    ax_cwt.grid(
        False
    )

    ax_cwt.set_xlim(
        1,
        len(scores)
    )

    #plt.tight_layout()

    plt.savefig(
        output_filename,
        dpi=300
    )

    # plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "machine_file",
        type=Path
    )

    parser.add_argument(
        "annotation_file",
        type=Path
    )

    parser.add_argument(
        "--filter",
        type=float,
        default=3,
        help="Размер окна медианного фильтра в строках"
    )

    parser.add_argument(
        "--cwt-min-scale",
        type=float,
        default=1
    )

    parser.add_argument(
        "--cwt-max-scale",
        type=float,
        default=50
    )

    parser.add_argument(
        "--cwt-scales",
        type=int,
        default=50
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # LOAD MACHINE DATA
    # ---------------------------------------------------------

    machine_data = load_machine_file(
        args.machine_file
    )

    scores = calculate_score(
        machine_data
    )

    # ---------------------------------------------------------
    # FILTER
    # ---------------------------------------------------------

    filtered_scores = nonlinear_median_filter(
        scores,
        args.filter
    )

    # ---------------------------------------------------------
    # CWT
    # ---------------------------------------------------------

    scales = np.linspace(
        args.cwt_min_scale,
        args.cwt_max_scale,
        args.cwt_scales
    )

    cwt, widths = calculate_cwt(
        filtered_scores,
        args.cwt_min_scale,
        args.cwt_max_scale
    )

    best_index = np.unravel_index(
        np.argmax(cwt),
        cwt.shape
    )

    best_width = widths[best_index[0]]
    best_center = best_index[1]

    det_start = max(
        1,
        best_center - best_width // 2 + 1
    )

    det_end = min(
        len(filtered_scores),
        det_start + best_width - 1
    )

    detected_bounds = [det_start, det_end]
   #detected_bounds = get_detected_bounds(
   #    active
   #)

    # ---------------------------------------------------------
    # LOAD ANNOTATION
    # ---------------------------------------------------------

    with open(
        args.annotation_file,
        "r",
        encoding="utf-8"
    ) as f:
        annotation_data = json.load(f)

    annotation_filename = args.annotation_file

    txt_filename = annotation_filename.with_suffix(
        ".txt"
    )

    with open(
        txt_filename,
        "r",
        encoding="utf-8"
    ) as f:
        text = f.read()

    annotations = get_reference_annotations(
        annotation_data
    )

    reference_bounds = get_reference_line_bounds(
        text,
        annotations
    )

    iou = calculate_iou(
        reference_bounds,
        detected_bounds
    )


    # ---------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------

    output_filename = args.machine_file.with_suffix(
        ".png"
    )

    make_plot(
        scores=scores,
        filtered_scores=filtered_scores,
        cwt=cwt,
        scales=scales,
        reference_bounds=reference_bounds,
        detected_bounds=detected_bounds,
        IoUvalue=iou,
        output_filename=output_filename
    )

    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------

    print()

    print(
        f"Reference: {reference_bounds}"
    )

    print(
        f"Detected : {detected_bounds}"
    )

    print(
        f"IoU      : {iou:.6f}"
    )

    print(
        f"CWT      : scales "
        f"{args.cwt_min_scale:.1f}–"
        f"{args.cwt_max_scale:.1f}"
    )

    print(
        f"Output   : {output_filename}"
    )


if __name__ == "__main__":
    main()