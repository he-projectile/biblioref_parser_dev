import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.ticker import MaxNLocator

from biblioBlockLocalization import localizeBiblioBlock


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
    return text[:char_pos].count("\n") + 1


def get_reference_line_bounds(text, annotations):
    if not annotations:
        return None

    starts = []
    ends = []

    for start, end in annotations:
        starts.append(char_to_line(text, start + 1))
        ends.append(char_to_line(text, end + 1))

    return min(starts), max(ends)


def load_reference_bounds(annotation_filename):
    if annotation_filename is None:
        return None

    annotation_filename = Path(annotation_filename)

    txt_filename = annotation_filename.with_suffix(".txt")

    with open(annotation_filename, "r", encoding="utf-8") as f:
        annotation_data = json.load(f)

    with open(txt_filename, "r", encoding="utf-8") as f:
        text = f.read()

    annotations = get_reference_annotations(annotation_data)

    return get_reference_line_bounds(text, annotations)


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

    ax_score = fig.add_subplot(gs[1, 0])
    ax_legend = fig.add_subplot(gs[1, 2])

    ax_legend.axis("off")

    return fig, ax_score, ax_legend


def add_reference(ax, reference_bounds):
    if reference_bounds is None:
        return

    ref_start, ref_end = reference_bounds

    plt.rcParams["hatch.linewidth"] = 0.20

    ax.axvspan(
        ref_start - 0.5,
        ref_end + 0.5,
        facecolor="none",
        edgecolor="black",
        hatch="//",
        linewidth=0.20,
        label=f"Эталон: строки {ref_start}–{ref_end}"
    )


def add_search_start(ax, search_start):
    # search_start внутри локализатора — индекс строки с нуля.
    # На графике строки нумеруются с единицы.
    search_start_line = search_start + 1

    ax.axvline(
        search_start_line,
        linestyle="--",
        linewidth=1.5,
        label=(
            f"Начало поиска "
            f"({search_start_line:.0f} строка)"
        )
    )


def add_legend(ax_legend, ax_score):
    handles, labels = ax_score.get_legend_handles_labels()

    ax_legend.legend(
        handles,
        labels,
        loc="center left"
    )


def setup_score_axis(ax, scores):
    x = np.arange(1, len(scores) + 1)

    ax.set_xlim(
        0.5,
        len(scores) + 0.5
    )

    # Только целые номера строк.
    # Количество подписей ограничиваем, чтобы при больших документах
    # ось не превращалась в кашу.
    ax.xaxis.set_major_locator(
        MaxNLocator(
            integer=True,
            nbins=15
        )
    )

    ax.set_xlabel("Номер строки")
    ax.set_ylabel("SCORE")

    ax.grid(
        True,
        alpha=0.3
    )

    return x


def make_plot_score(
    scores,
    reference_bounds,
    output_filename,
    source_filename
):
    fig, ax_score, ax_legend = make_base_figure()

    x = setup_score_axis(
        ax_score,
        scores
    )

    # ВАЖНО:
    # оставляем именно step(), как в исходном plotBiblioBlockLocalization.py
    ax_score.step(
        x,
        scores,
        linewidth=1.0,
        where="mid",
        label="SCORE"
    )

    add_reference(
        ax_score,
        reference_bounds
    )

    ax_score.set_title(
        "SCORE"
    )

    add_legend(
        ax_legend,
        ax_score
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


def make_plot_score_filtered(
    scores,
    filtered_scores,
    reference_bounds,
    output_filename,
    source_filename
):
    fig, ax_score, ax_legend = make_base_figure()

    x = setup_score_axis(
        ax_score,
        scores
    )

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
        label="SCORE после фильтра длины"
    )

    add_reference(
        ax_score,
        reference_bounds
    )

    ax_score.set_title(
        "SCORE и результат фильтрации"
    )

    add_legend(
        ax_legend,
        ax_score
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


def make_plot_score_filtered_search(
    scores,
    filtered_scores,
    search_start,
    reference_bounds,
    output_filename,
    source_filename
):
    fig, ax_score, ax_legend = make_base_figure()

    x = setup_score_axis(
        ax_score,
        scores
    )

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
        label="SCORE после фильтра длины"
    )

    add_search_start(
        ax_score,
        search_start
    )

    add_reference(
        ax_score,
        reference_bounds
    )

    ax_score.set_title(
        "SCORE, результат фильтрации и начало поиска"
    )

    add_legend(
        ax_legend,
        ax_score
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

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    source_filename = (
        args.machine_file
        .with_suffix(".txt")
        .name
    )

    result = localizeBiblioBlock(
        args.machine_file,
        args.patterns
    )

    scores = result["scores"]
    filtered_scores = result["filtered_scores"]
    search_start = result["search_start"]

    reference_bounds = load_reference_bounds(
        args.annotation_file
    )

    stem = args.machine_file.stem

    score_filename = (
        args.output_dir
        / f"{stem}_score.png"
    )

    filtered_filename = (
        args.output_dir
        / f"{stem}_score_filtered.png"
    )

    search_filename = (
        args.output_dir
        / f"{stem}_score_filtered_search.png"
    )

    # ---------------------------------------------------------
    # 1. SCORE + эталон
    # ---------------------------------------------------------

    make_plot_score(
        scores=scores,
        reference_bounds=reference_bounds,
        output_filename=score_filename,
        source_filename=source_filename
    )

    # ---------------------------------------------------------
    # 2. SCORE + filtered + эталон
    # ---------------------------------------------------------

    make_plot_score_filtered(
        scores=scores,
        filtered_scores=filtered_scores,
        reference_bounds=reference_bounds,
        output_filename=filtered_filename,
        source_filename=source_filename
    )

    # ---------------------------------------------------------
    # 3. SCORE + filtered + начало поиска + эталон
    # ---------------------------------------------------------

    make_plot_score_filtered_search(
        scores=scores,
        filtered_scores=filtered_scores,
        search_start=search_start,
        reference_bounds=reference_bounds,
        output_filename=search_filename,
        source_filename=source_filename
    )

    print()
    print(f"Источник: {source_filename}")
    print(f"Эталонные границы: {reference_bounds}")
    print(f"Начало поиска: строка {search_start + 1}")
    print()
    print(f"1. SCORE:                 {score_filename}")
    print(f"2. SCORE + filtered:      {filtered_filename}")
    print(f"3. SCORE + filtered + search: {search_filename}")


if __name__ == "__main__":
    main()