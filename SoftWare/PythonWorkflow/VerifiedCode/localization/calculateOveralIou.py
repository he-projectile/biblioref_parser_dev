import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from biblioBlockLocalization import localizeBiblioBlock


REFERENCE_LABEL = "БИБЛ. ССЫЛКА"


def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


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

        start_line = char_to_line(
            text,
            start + 1
        )

        end_line = char_to_line(
            text,
            end + 1
        )

        starts.append(start_line)
        ends.append(end_line)

    return min(starts), max(ends)


def calculate_iou(reference_bounds, detected_bounds):
    if reference_bounds is None:
        return 0.0

    if detected_bounds is None:
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
    machine_file,
    text_file,
    annotation_file,
    patterns_file
):
    machine_file = Path(machine_file)
    text_file = Path(text_file)
    annotation_file = Path(annotation_file)
    patterns_file = Path(patterns_file)

    # ---------------------------------------------------------
    # TXT
    # ---------------------------------------------------------

    with open(
        text_file,
        "r",
        encoding="utf-8"
    ) as f:
        text = f.read()

    # ---------------------------------------------------------
    # ANNOTATION
    # ---------------------------------------------------------

    annotation_data = load_json(
        annotation_file
    )

    annotations = get_reference_annotations(
        annotation_data
    )

    reference_bounds = get_reference_line_bounds(
        text,
        annotations
    )

    # ---------------------------------------------------------
    # LOCALIZATION
    # ---------------------------------------------------------

    result = localizeBiblioBlock(
        machine_file,
        patterns_file
    )

    if result["start"] is None:
        detected_bounds = None
    else:
        detected_bounds = (
            result["start"],
            result["end"]
        )

    # ---------------------------------------------------------
    # IoU
    # ---------------------------------------------------------

    iou = calculate_iou(
        reference_bounds,
        detected_bounds
    )

    return {
        "filename": machine_file.stem[len("MACHINE_"):],
        "reference": reference_bounds,
        "detected": detected_bounds,
        "iou": iou
    }


def make_histogram(results, output_filename):
    ious = np.array(
        [
            result["iou"]
            for result in results
        ],
        dtype=float
    )

    if len(ious) == 0:
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

    plt.xlim(
        0.0,
        1.0
    )

    plt.grid(
        True,
        axis="y",
        alpha=0.3
    )

    plt.text(
        0.02,
        0.95,
        f"Среднее IoU = {mean_iou:.4f}\n"
        f"Медиана IoU = {median_iou:.4f}\n"
        f"N = {len(ious)}",
        transform=plt.gca().transAxes,
        ha="left",
        va="top"
    )

    ax = plt.gca()  # Получаем текущую ось
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Делаем тики целыми

    plt.tight_layout()

    plt.savefig(
        output_filename,
        dpi=300
    )

    plt.close()

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Calculate IoU for bibliography "
            "block localization."
        )
    )

    parser.add_argument(
        "--texts",
        type=Path,
        required=True,
        help="Directory with TXT files"
    )

    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Directory with annotation JSON files"
    )

    parser.add_argument(
        "--machine",
        type=Path,
        required=True,
        help="Directory with MACHINE_*.json files"
    )

    parser.add_argument(
        "--patterns",
        type=Path,
        required=True,
        help="patterns.json"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Histogram output filename"
    )

    args = parser.parse_args()

    text_dir = args.texts
    annotation_dir = args.annotations
    machine_dir = args.machine
    patterns_file = args.patterns

    print(f"Machine dir: {machine_dir}")
    print(f"Absolute path: {machine_dir.resolve()}")
    print(f"Exists: {machine_dir.exists()}")
    print(f"Is directory: {machine_dir.is_dir()}")

    print("Files in directory:")
    for f in machine_dir.iterdir():
        print(f"  {f.name}")

    machine_files = sorted(machine_dir.glob("MACHINE_*.json"))

    print(f"Matched MACHINE files: {len(machine_files)}")
    for f in machine_files:
        print(f"  {f}")

    print(
        f"Found {len(machine_files)} MACHINE files."
    )

    if not machine_files:
        return

    results = []

    for machine_file in machine_files:

        stem = machine_file.stem[
            len("MACHINE_"):
        ]

        text_file = (
            text_dir / f"{stem}.txt"
        )

        annotation_file = (
            annotation_dir / f"{stem}.json"
        )

        if not text_file.exists():
            print(
                f"WARNING: TXT not found: "
                f"{text_file}"
            )
            continue

        if not annotation_file.exists():
            print(
                f"WARNING: annotation not found: "
                f"{annotation_file}"
            )
            continue

        try:

            result = process_file(
                machine_file=machine_file,
                text_file=text_file,
                annotation_file=annotation_file,
                patterns_file=patterns_file
            )

            results.append(result)

            print(
                f"{result['filename']}: "
                f"reference={result['reference']}, "
                f"detected={result['detected']}, "
                f"IoU={result['iou']:.6f}"
            )

        except Exception as e:

            print(
                f"ERROR: "
                f"{machine_file.name}: {e}"
            )

    if not results:
        print("No files were processed.")
        return

    # ---------------------------------------------------------
    # STATISTICS
    # ---------------------------------------------------------

    ious = np.array(
        [
            result["iou"]
            for result in results
        ],
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
            machine_dir / "IoU_histogram.png"
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