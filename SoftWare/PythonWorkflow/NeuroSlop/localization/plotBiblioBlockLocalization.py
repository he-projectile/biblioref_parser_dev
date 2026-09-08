import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from biblioBlockLocalization import (
    localizeBiblioBlock,
    calculate_iou,
    load_reference_bounds,
)


def plot_localization(
    machine_filename,
    patterns_filename,
    output_filename,
    annotation_filename=None,
):
    """
    Локализация библиографического блока и построение графика.

    Если annotation_filename указан, на графике дополнительно
    отображается эталонный библиографический блок и рассчитывается IoU.
    """

    # ---------------------------------------------------------
    # Локализация
    # ---------------------------------------------------------

    result = localizeBiblioBlock(
        machine_filename,
        patterns_filename
    )

    scores = np.asarray(result["scores"])
    filtered_scores = np.asarray(result["filtered_scores"])

    detected_start = result["start"]
    detected_end = result["end"]

    lines = np.arange(1, len(scores) + 1)

    # ---------------------------------------------------------
    # Эталонная разметка
    # ---------------------------------------------------------

    reference_start = None
    reference_end = None
    iou = None

    if annotation_filename is not None:
        reference_bounds = load_reference_bounds(
            annotation_filename
        )

        if reference_bounds is not None:
            reference_start, reference_end = reference_bounds

            detected_bounds = (
                detected_start,
                detected_end
            )

            iou = calculate_iou(
                reference_bounds,
                detected_bounds
            )

    # ---------------------------------------------------------
    # График
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(15, 7)
    )

    # ---------------------------------------------------------
    # Исходный score
    # ---------------------------------------------------------

    ax.step(
        lines,
        scores,
        where="mid",
        linewidth=1.2,
        label="Score"
    )

    # ---------------------------------------------------------
    # Score после медианного фильтра
    # ---------------------------------------------------------

    ax.step(
        lines,
        filtered_scores,
        where="mid",
        linewidth=2,
        label="Filtered score"
    )

    # ---------------------------------------------------------
    # Диапазон найденного блока
    #
    # Штриховка /
    # ---------------------------------------------------------

    y_min, y_max = ax.get_ylim()

    ax.fill_between(
        lines,
        0,
        filtered_scores,
        where=(
            (lines >= detected_start) &
            (lines <= detected_end)
        ),
        step="mid",
        alpha=0.25,
        hatch="///",
        facecolor="none",
        edgecolor="black",
        linewidth=0.0,
        label=(
            f"Detected "
            f"[{detected_start}, {detected_end}]"
        )
    )

    # ---------------------------------------------------------
    # Эталонный блок
    #
    # Штриховка \
    # ---------------------------------------------------------

    if (
        reference_start is not None
        and reference_end is not None
    ):
        ax.fill_between(
            lines,
            0,
            filtered_scores,
            where=(
                (lines >= reference_start) &
                (lines <= reference_end)
            ),
            step="mid",
            alpha=0.25,
            hatch="\\\\\\",
            facecolor="none",
            edgecolor="black",
            linewidth=0.0,
            label=(
                f"Reference "
                f"[{reference_start}, {reference_end}]"
            )
        )

    # ---------------------------------------------------------
    # Вертикальные границы найденного блока
    # ---------------------------------------------------------

    ax.axvline(
        detected_start,
        linestyle="--",
        linewidth=1
    )

    ax.axvline(
        detected_end,
        linestyle="--",
        linewidth=1
    )

    # ---------------------------------------------------------
    # Вертикальные границы эталонного блока
    # ---------------------------------------------------------

    if (
        reference_start is not None
        and reference_end is not None
    ):
        ax.axvline(
            reference_start,
            linestyle=":",
            linewidth=1
        )

        ax.axvline(
            reference_end,
            linestyle=":",
            linewidth=1
        )

    # ---------------------------------------------------------
    # IoU
    # ---------------------------------------------------------

    if iou is not None:
        ax.text(
            0.02,
            0.95,
            f"IoU = {iou:.4f}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=12
        )

    # ---------------------------------------------------------
    # Оформление
    # ---------------------------------------------------------

    ax.set_xlabel("Line")
    ax.set_ylabel("Score")

    ax.set_title(
        "Bibliographic block localization"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    # ---------------------------------------------------------
    # Сохранение
    # ---------------------------------------------------------

    fig.savefig(
        output_filename,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    return {
        "detected_start": detected_start,
        "detected_end": detected_end,
        "reference_start": reference_start,
        "reference_end": reference_end,
        "iou": iou,
    }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Plot bibliographic block localization"
        )
    )

    parser.add_argument(
        "machine",
        help="MACHINE_*.json file"
    )

    parser.add_argument(
        "--patterns",
        required=True,
        help="Patterns JSON file"
    )

    parser.add_argument(
        "--annotation",
        default=None,
        help="Annotation JSON file"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for output plots"
    )

    args = parser.parse_args()

    machine_path = Path(args.machine)
    patterns_path = Path(args.patterns)
    output_dir = Path(args.output_dir)

    annotation_path = None

    if args.annotation is not None:
        annotation_path = Path(args.annotation)

    # ---------------------------------------------------------
    # Проверки
    # ---------------------------------------------------------

    if not machine_path.exists():
        raise FileNotFoundError(
            f"MACHINE file not found: {machine_path}"
        )

    if not patterns_path.exists():
        raise FileNotFoundError(
            f"Patterns file not found: {patterns_path}"
        )

    if annotation_path is not None:
        if not annotation_path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: "
                f"{annotation_path}"
            )

    # ---------------------------------------------------------
    # Создание выходной папки
    # ---------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Имя выходного файла
    #
    # MACHINE_article.json
    #       ↓
    # article.png
    # ---------------------------------------------------------

    stem = machine_path.stem

    if stem.startswith("MACHINE_"):
        stem = stem[len("MACHINE_"):]

    output_filename = (
        output_dir / f"{stem}.png"
    )

    # ---------------------------------------------------------
    # Построение
    # ---------------------------------------------------------

    result = plot_localization(
        machine_filename=machine_path,
        patterns_filename=patterns_path,
        output_filename=output_filename,
        annotation_filename=annotation_path,
    )

    # ---------------------------------------------------------
    # Вывод результата
    # ---------------------------------------------------------

    print(f"Machine:   {machine_path}")
    print(f"Patterns:  {patterns_path}")

    if annotation_path is not None:
        print(f"Annotation: {annotation_path}")

    print(f"Output:    {output_filename}")
    print()

    print(
        f"Detected block: "
        f"{result['detected_start']} - "
        f"{result['detected_end']}"
    )

    if result["reference_start"] is not None:
        print(
            f"Reference block: "
            f"{result['reference_start']} - "
            f"{result['reference_end']}"
        )

        print(
            f"IoU: {result['iou']:.4f}"
        )


if __name__ == "__main__":
    main()
