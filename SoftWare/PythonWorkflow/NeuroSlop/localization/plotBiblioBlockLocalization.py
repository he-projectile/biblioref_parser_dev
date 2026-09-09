import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from biblioBlockLocalization import (
    localizeBiblioBlock,
)


REFERENCE_LABEL = "БИБЛ. ССЫЛКА"


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

    return (
        min(starts),
        max(ends)
    )


def load_reference_bounds(annotation_filename):

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

    fig = plt.figure(
        figsize=(20, 12),
        dpi=300
    )

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

    ax_cwt = fig.add_subplot(
        gs[0, 0]
    )

    ax_score = fig.add_subplot(
        gs[1, 0],
        sharex=ax_cwt
    )

    ax_cbar = fig.add_subplot(
        gs[0, 1]
    )

    ax_legend = fig.add_subplot(
        gs[1, 2]
    )

    ax_legend.axis("off")

    x = np.arange(
        1,
        len(scores) + 1
    )

    # --------------------------------------------------
    # CWT heatmap
    # --------------------------------------------------

    max_abs = np.max(
        np.abs(cwt)
    )

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

    ax_cwt.set_ylabel(
        "Ширина окна, строк"
    )

    ax_cwt.set_title(
        "CWT прямоугольного окна"
    )

    fig.colorbar(
        image,
        cax=ax_cbar,
        label="Интенсивность CWT"
    )

    # --------------------------------------------------
    # Score
    # --------------------------------------------------

    ax_score.step(
        x,
        scores,
        linewidth=1.0,
        where="mid",
        label="SCORE"
    )

    ax_score.step(
        x,
        filtered_scores,
        linewidth=2.0,
        where="mid",
        label="SCORE после нелинейного ФНЧ"
    )

    # --------------------------------------------------
    # Reference
    # --------------------------------------------------

    if reference_bounds is not None:

        ref_start, ref_end = reference_bounds

        plt.rcParams[
            "hatch.linewidth"
        ] = 0.20

        ax_score.axvspan(
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

    # --------------------------------------------------
    # Detection
    # --------------------------------------------------

    if detected_bounds is not None:

        det_start, det_end = detected_bounds

        plt.rcParams[
            "hatch.linewidth"
        ] = 0.20

        ax_score.axvspan(
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

    # --------------------------------------------------
    # Axes
    # --------------------------------------------------

    ax_score.set_xlabel(
        "Номер строки"
    )

    ax_score.set_ylabel(
        "SCORE"
    )

    ax_score.set_title(
        "SCORE и результат нелинейного обнаружения"
    )

    ax_score.grid(
        True,
        alpha=0.3
    )

    # --------------------------------------------------
    # Legend
    # --------------------------------------------------

    handles, labels = (
        ax_score.get_legend_handles_labels()
    )

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

    ax_cwt.grid(False)

    ax_cwt.set_xlim(
        1,
        len(scores)
    )

    plt.savefig(
        output_filename,
        dpi=300
    )

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
        "--patterns",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".")
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # Localization
    # --------------------------------------------------

    result = localizeBiblioBlock(
        args.machine_file,
        args.patterns
    )

    scores = result["scores"]

    filtered_scores = result[
        "filtered_scores"
    ]

    cwt = result["cwt"]

    widths = result["widths"]

    detected_bounds = None

    if (
        result["start"] is not None
        and result["end"] is not None
    ):
        detected_bounds = (
            result["start"],
            result["end"]
        )

    # --------------------------------------------------
    # Reference
    # --------------------------------------------------

    reference_bounds = load_reference_bounds(
        args.annotation_file
    )

    # --------------------------------------------------
    # IoU
    # --------------------------------------------------

    IoUvalue = calculate_iou(
        reference_bounds,
        detected_bounds
    )

    # --------------------------------------------------
    # Output
    # --------------------------------------------------

    output_filename = (
        args.output_dir
        / args.machine_file.with_suffix(
            ".png"
        ).name
    )

    make_plot(
        scores,
        filtered_scores,
        cwt,
        widths,
        reference_bounds,
        detected_bounds,
        IoUvalue,
        output_filename
    )

    print(
        f"Эталон: "
        f"{reference_bounds}"
    )

    print(
        f"Распознано: "
        f"{detected_bounds}"
    )

    print(
        f"IoU = {IoUvalue:.4f}"
    )

    print(
        f"График сохранён: "
        f"{output_filename}"
    )


if __name__ == "__main__":
    main()