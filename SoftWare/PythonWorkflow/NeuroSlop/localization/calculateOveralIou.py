import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


REFERENCE_LABEL = "БИБЛ. ССЫЛКА"


def load_machine_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_score(machine_data):
    """
    Использует уже рассчитанный score из MACHINE_*.json.
    """

    lines = machine_data["lines"]

    scores = []

    for line in lines:
        scores.append(line["score"])

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

    return before.count("\n") + 1


def get_reference_line_bounds(text, annotations):
    if not annotations:
        return None

    starts = []
    ends = []

    for start, end in annotations:
        start_line = char_to_line(text, start + 1)
        end_line = char_to_line(text, end + 1)

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

        padded_signal = (
            padded_signal
            - np.mean(padded_signal)*1.5
        )

        values = np.convolve(
            padded_signal,
            kernel,
            mode="valid"
        )

        result[i] = values

    return result, widths


def calculate_detected_bounds(
    scores,
    filter_size=3,
    cwt_min_scale=1,
    cwt_max_scale=75
):
    """
    Полностью повторяет логику обнаружения
    из plotRecognition.py.
    """

    filtered_scores = nonlinear_median_filter(
        scores,
        filter_size
    )

    cwt, widths = calculate_cwt(
        filtered_scores,
        cwt_min_scale,
        cwt_max_scale
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

    return [det_start, det_end]


def calculate_iou(reference_bounds, detected_bounds):
    if reference_bounds is None or detected_bounds is None:
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


def process_file(
    machine_filename,
    filter_size,
    cwt_min_scale,
    cwt_max_scale
):
    machine_filename = Path(machine_filename)

    # ---------------------------------------------------------
    # MACHINE
    # ---------------------------------------------------------

    machine_data = load_machine_file(
        machine_filename
    )

    scores = calculate_score(
        machine_data
    )

    # ---------------------------------------------------------
    # DETECTION
    # ---------------------------------------------------------

    detected_bounds = calculate_detected_bounds(
        scores,
        filter_size=filter_size,
        cwt_min_scale=cwt_min_scale,
        cwt_max_scale=cwt_max_scale
    )

    # ---------------------------------------------------------
    # ANNOTATION
    # ---------------------------------------------------------

    filename = machine_filename.name

    if not filename.startswith("MACHINE_"):
        print(
            f"WARNING: unexpected filename: {filename}"
        )
        return None

    stem = machine_filename.stem[len("MACHINE_"):]

    annotation_filename = (
        machine_filename.parent / f"{stem}.json"
    )

    if not annotation_filename.exists():
        print(
            f"WARNING: annotation not found: "
            f"{annotation_filename}"
        )
        return None

    with open(
        annotation_filename,
        "r",
        encoding="utf-8"
    ) as f:
        annotation_data = json.load(f)

    # ---------------------------------------------------------
    # TXT
    # ---------------------------------------------------------

    txt_filename = (
        machine_filename.parent / f"{stem}.txt"
    )

    if not txt_filename.exists():
        print(
            f"WARNING: TXT not found: "
            f"{txt_filename}"
        )
        return None

    with open(
        txt_filename,
        "r",
        encoding="utf-8"
    ) as f:
        text = f.read()

    # ---------------------------------------------------------
    # REFERENCE
    # ---------------------------------------------------------

    annotations = get_reference_annotations(
        annotation_data
    )

    reference_bounds = get_reference_line_bounds(
        text,
        annotations
    )

    # ---------------------------------------------------------
    # IoU
    # ---------------------------------------------------------

    iou = calculate_iou(
        reference_bounds,
        detected_bounds
    )

    return {
        "filename": stem,
        "reference": reference_bounds,
        "detected": detected_bounds,
        "iou": iou
    }


def make_histogram(results, output_filename):
    ious = [
        result["iou"]
        for result in results
    ]

    if not ious:
        print("No IoU values to plot.")
        return

    mean_iou = np.mean(ious)
    median_iou = np.median(ious)

    plt.figure(
        figsize=(10, 6),
        dpi=150
    )

    plt.hist(
        ious,
        bins=20,
        range=(0.0, 1.0),
        edgecolor="black"
    )

    plt.xlabel("IoU")
    plt.ylabel("Количество документов")

    plt.title(
        "Распределение IoU локализации "
        "библиографического блока"
    )

    plt.xlim(0.0, 1.0)

    plt.grid(
        True,
        axis="y",
        alpha=0.3
    )

    plt.text(
        0.98,
        0.95,
        f"Среднее IoU = {mean_iou:.4f}\n"
        f"Медиана IoU = {median_iou:.4f}\n"
        f"N = {len(ious)}",
        transform=plt.gca().transAxes,
        ha="right",
        va="top"
    )

    plt.tight_layout()

    plt.savefig(
        output_filename,
        dpi=300
    )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate IoU for all MACHINE_*.json "
            "files and build IoU histogram."
        )
    )

    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing MACHINE_*.json files"
    )

    parser.add_argument(
        "--filter",
        type=float,
        default=3,
        help="Median filter window size"
    )

    parser.add_argument(
        "--cwt-min-scale",
        type=int,
        default=1
    )

    parser.add_argument(
        "--cwt-max-scale",
        type=int,
        default=75
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Histogram output filename"
    )

    args = parser.parse_args()

    directory = args.directory

    machine_files = sorted(
        directory.glob("MACHINE_*.json")
    )

    print(
        f"Found {len(machine_files)} MACHINE files."
    )

    if not machine_files:
        return

    results = []

    for machine_file in machine_files:

        try:
            result = process_file(
                machine_file,
                filter_size=args.filter,
                cwt_min_scale=args.cwt_min_scale,
                cwt_max_scale=args.cwt_max_scale
            )

            if result is None:
                continue

            results.append(result)

            print(
                f"{result['filename']}: "
                f"reference={result['reference']}, "
                f"detected={result['detected']}, "
                f"IoU={result['iou']:.6f}"
            )

        except Exception as e:
            print(
                f"ERROR: {machine_file.name}: {e}"
            )

    if not results:
        print("No files were processed.")
        return

    # ---------------------------------------------------------
    # STATISTICS
    # ---------------------------------------------------------

    ious = np.array(
        [result["iou"] for result in results],
        dtype=float
    )

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(
        f"Documents : {len(results)}"
    )

    print(
        f"Mean IoU  : {np.mean(ious):.6f}"
    )

    print(
        f"Median IoU: {np.median(ious):.6f}"
    )

    print(
        f"Min IoU   : {np.min(ious):.6f}"
    )

    print(
        f"Max IoU   : {np.max(ious):.6f}"
    )

    print(
        f"IoU >= 0.5: "
        f"{np.sum(ious >= 0.5)} / {len(ious)}"
    )

    print(
        f"IoU >= 0.9: "
        f"{np.sum(ious >= 0.9)} / {len(ious)}"
    )

    print(
        f"IoU >= 0.95: "
        f"{np.sum(ious >= 0.95)} / {len(ious)}"
    )

    # ---------------------------------------------------------
    # HISTOGRAM
    # ---------------------------------------------------------

    if args.output is None:
        output_filename = (
            directory / "IoU_histogram.png"
        )
    else:
        output_filename = args.output

    make_histogram(
        results,
        output_filename
    )

    print()
    print(
        f"Histogram: {output_filename}"
    )


if __name__ == "__main__":
    main()