import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

from biblioBlockLocalization import localizeBiblioBlockData, loadLocalizationData


# ============================================================
# Configuration
# ============================================================

DEFAULT_SEED = 42

WEIGHT_MIN = -10.0
WEIGHT_MAX = 10.0

POP_SIZE = 1
MAX_ITER = 1
TOL = 1e-7

REFERENCE_LABEL = "БИБЛ. ССЫЛКА"


# ============================================================
# JSON
# ============================================================

def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Annotation processing
# ============================================================

def get_reference_annotations(annotation_data):
    """
    Recursively finds all annotations with label:
        БИБЛ. ССЫЛКА

    Returns:
        [(start_char, end_char), ...]
    """

    result = []

    def recursive_search(obj):

        if isinstance(obj, dict):

            if (
                obj.get("label") == REFERENCE_LABEL
                and "start" in obj
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

    recursive_search(annotation_data)

    return result


def char_to_line(text, char_pos):
    """
    Converts character position to 1-based line number.
    """

    return text[:char_pos].count("\n") + 1


def get_reference_line_bounds(text, annotations):
    """
    Converts bibliography reference character spans
    into one enclosing line interval.

    Returns:
        (start_line, end_line)

    or:
        None
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

        starts.append(start_line)
        ends.append(end_line)

    return min(starts), max(ends)


# ============================================================
# IoU
# ============================================================

def calculate_iou(reference_bounds, detected_bounds):
    """
    Calculates IoU between two inclusive line intervals.
    """

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


# ============================================================
# Dataset
# ============================================================

def prepare_dataset(
    texts_dir,
    annotations_dir,
    machine_dir
):
    texts_dir = Path(texts_dir)
    annotations_dir = Path(annotations_dir)
    machine_dir = Path(machine_dir)

    dataset = []

    txt_files = sorted(
        texts_dir.glob("*.txt")
    )

    for text_file in txt_files:

        stem = text_file.stem

        annotation_file = (
            annotations_dir / f"{stem}.json"
        )

        machine_file = (
            machine_dir / f"MACHINE_{stem}.json"
        )

        if not annotation_file.exists():
            print(
                f"[WARNING] Нет annotation: "
                f"{annotation_file}"
            )
            continue

        if not machine_file.exists():
            print(
                f"[WARNING] Нет MACHINE: "
                f"{machine_file}"
            )
            continue

        with open(
            text_file,
            "r",
            encoding="utf-8"
        ) as f:
            text = f.read()

        annotation_data = load_json(
            annotation_file
        )

        annotations = get_reference_annotations(
            annotation_data
        )

        target = get_reference_line_bounds(
            text,
            annotations
        )

        if target is None:
            print(
                f"[WARNING] Нет БИБЛ. ССЫЛКА: "
                f"{text_file}"
            )
            continue

        localization_data = loadLocalizationData(
            machine_file
        )

        dataset.append(
            {
                "name": stem,
                "text_file": str(text_file),
                "annotation_file": str(annotation_file),
                "machine_file": str(machine_file),
                "localization_data": localization_data,
                "target": target
            }
        )

    return dataset


# ============================================================
# Dataset split
# ============================================================

def split_dataset(
    dataset,
    train_ratio=0.7,
    validation_ratio=0.15,
    seed=DEFAULT_SEED
):
    """
    Splits dataset into:

        train
        validation
        test
    """

    dataset = list(dataset)

    rng = random.Random(seed)
    rng.shuffle(dataset)

    n = len(dataset)

    train_end = int(
        n * train_ratio
    )

    validation_end = train_end + int(
        n * validation_ratio
    )

    train = dataset[:train_end]

    validation = dataset[
        train_end:validation_end
    ]

    test = dataset[
        validation_end:
    ]

    return train, validation, test


# ============================================================
# IoU evaluation
# ============================================================

def evaluate_weights(
    dataset,
    weights
):
    """
    Runs the bibliography localizer on the whole dataset
    using the supplied weight vector.

    Returns:

        mean_iou
        ious
    """

    if len(dataset) == 0:
        return 0.0, []

    ious = []

    for document in dataset:

        try:

            result = localizeBiblioBlockData(
                document["localization_data"],
                weights
            )

            if result["start"] is None:
                detected_bounds = None
            else:
                detected_bounds = (
                    result["start"],
                    result["end"]
                )

            iou = calculate_iou(
                document["target"],
                detected_bounds
            )

            ious.append(iou)

        except Exception as e:

            print(
                f"\n[WARNING] Localization failed "
                f"for {document['name']}: {e}"
            )

            ious.append(0.0)

    mean_iou = float(
        np.mean(ious)
    )

    return mean_iou, ious


# ============================================================
# Optimizer
# ============================================================

def optimize(
    train,
    patterns_filename,
    pattern_count,
    seed=DEFAULT_SEED
):
    """
    Optimizes pattern weights using
    scipy.optimize.differential_evolution.

    Objective:

        error = -mean(IoU)

    """

    dimension = pattern_count

    bounds = [
        (WEIGHT_MIN, WEIGHT_MAX)
        for _ in range(dimension)
    ]

    iteration_history = []

    evaluation_counter = 0
    best_iou = -1.0

    def objective(weights):

        nonlocal evaluation_counter
        nonlocal best_iou

        evaluation_counter += 1

        mean_iou, _ = evaluate_weights(
            train,
            weights
        )

        if mean_iou > best_iou:

            best_iou = mean_iou

            print(
                f"\nNEW BEST | "
                f"evaluation {evaluation_counter} | "
                f"mean IoU = {mean_iou:.6f}",
                flush=True
            )

        elif evaluation_counter % 10 == 0:

            print(
                f"Evaluation {evaluation_counter} | "
                f"mean IoU = {mean_iou:.6f} | "
                f"best = {best_iou:.6f}",
                flush=True
            )

        return -mean_iou

    def callback(xk, convergence):

        mean_iou, _ = evaluate_weights(
            train,
            xk
        )

        iteration_number = (
            len(iteration_history) + 1
        )

        iteration_history.append(
            mean_iou
        )

        print(
            f"Iteration "
            f"{iteration_number:3d}/{MAX_ITER} | "
            f"mean IoU = {mean_iou:.6f}"
        )

        return False

    print()
    print("=" * 70)
    print("Starting differential evolution")
    print("=" * 70)

    print(
        f"Patterns: {pattern_count}"
    )

    print(
        f"Training documents: {len(train)}"
    )

    print(
        f"Weight bounds: "
        f"[{WEIGHT_MIN}, {WEIGHT_MAX}]"
    )

    print(
        f"Population size: {POP_SIZE}"
    )

    print(
        f"Maximum iterations: {MAX_ITER}"
    )

    print(
        f"Seed: {seed}"
    )

    print("=" * 70)
    print()

    result = differential_evolution(
        objective,
        bounds,
        seed=seed,
        popsize=POP_SIZE,
        maxiter=MAX_ITER,
        tol=TOL,
        polish=False,
        workers=1,
        updating="immediate",
        disp=False,
        callback=callback
    )

    # In case scipy finishes before callback is called
    # for the final solution.
    final_iou, _ = evaluate_weights(
        train,
        result.x
    )

    if (
        not iteration_history
        or abs(
            iteration_history[-1]
            - final_iou
        ) > 1e-12
    ):
        iteration_history.append(
            final_iou
        )

    print()
    print("=" * 70)
    print("Optimization finished")
    print("=" * 70)

    print(
        f"Best mean IoU: "
        f"{final_iou:.6f}"
    )

    print(
        f"Function evaluations: "
        f"{result.nfev}"
    )

    print(
        f"Iterations: "
        f"{result.nit}"
    )

    print("=" * 70)
    print()

    return result, iteration_history


# ============================================================
# Save optimized patterns
# ============================================================

def save_optimized_patterns(
    patterns_filename,
    weights,
    output_filename
):
    """
    Saves patterns.json with optimized weights.
    """

    data = load_json(
        patterns_filename
    )

    patterns = data["patterns"]

    if len(patterns) != len(weights):

        raise ValueError(
            f"Pattern count "
            f"({len(patterns)}) does not match "
            f"weight count ({len(weights)})"
        )

    for pattern, weight in zip(
        patterns,
        weights
    ):
        pattern["weight"] = float(weight)

    with open(
        output_filename,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4
        )


# ============================================================
# Save optimization history
# ============================================================

def save_history(
    history,
    output_filename
):
    data = {
        "iterations": list(
            range(
                1,
                len(history) + 1
            )
        ),
        "mean_iou": [
            float(x)
            for x in history
        ]
    }

    with open(
        output_filename,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4
        )


# ============================================================
# Plot IoU
# ============================================================

def plot_iou_history(
    history,
    output_filename
):
    """
    Plots mean IoU versus optimization iteration.
    """

    if not history:
        return

    iterations = np.arange(
        1,
        len(history) + 1
    )

    fig, ax = plt.subplots(
        figsize=(10, 6),
        dpi=150
    )

    ax.plot(
        iterations,
        history,
        linewidth=2.0,
        marker="o",
        markersize=3
    )

    ax.set_xlabel(
        "Итерация"
    )

    ax.set_ylabel(
        "Средний IoU"
    )

    ax.set_title(
        "Изменение среднего IoU в процессе оптимизации"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.set_xlim(
        1,
        len(history)
    )

    ax.set_ylim(
        0.0,
        1.0
    )

    fig.tight_layout()

    fig.savefig(
        output_filename,
        dpi=300
    )

    plt.close(fig)


# ============================================================
# Print weights
# ============================================================

def print_optimized_weights(
    patterns_filename,
    weights
):
    patterns_data = load_json(
        patterns_filename
    )

    patterns = patterns_data["patterns"]

    print()
    print("=" * 70)
    print("Optimized weights")
    print("=" * 70)

    for i, (pattern, weight) in enumerate(
        zip(patterns, weights)
    ):

        print(
            f"{i:3d} | "
            f"{weight: .6f} | "
            f"{pattern['name']}"
        )

    print("=" * 70)
    print()


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Optimize bibliography localization "
            "pattern weights using differential evolution."
        )
    )

    parser.add_argument(
        "--texts",
        required=True
    )

    parser.add_argument(
        "--annotations",
        required=True
    )

    parser.add_argument(
        "--machine",
        required=True
    )

    parser.add_argument(
        "--patterns",
        required=True,
        help="patterns.json"
    )

    parser.add_argument(
        "--output-patterns",
        default="patterns_optimized.json",
        help=(
            "Output file for optimized patterns "
            "(default: patterns_optimized.json)"
        )
    )

    parser.add_argument(
        "--plot",
        default="optimization_iou.png",
        help=(
            "Output IoU plot "
            "(default: optimization_iou.png)"
        )
    )

    parser.add_argument(
        "--history",
        default="optimization_history.json",
        help=(
            "Output optimization history "
            "(default: optimization_history.json)"
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            f"Random seed "
            f"(default: {DEFAULT_SEED})"
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Random seeds
    # --------------------------------------------------------

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print(
        "Preparing dataset..."
    )

    dataset = prepare_dataset(
        args.texts,
        args.annotations,
        args.machine
    )

    if not dataset:

        raise RuntimeError(
            "Dataset is empty."
        )

    print(
        f"Found {len(dataset)} documents."
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train, validation, test = split_dataset(
        dataset,
        seed=args.seed
    )

    print(
        f"Train:      {len(train)}"
    )

    print(
        f"Validation: {len(validation)}"
    )

    print(
        f"Test:       {len(test)}"
    )

    # --------------------------------------------------------
    # Patterns
    # --------------------------------------------------------

    patterns_data = load_json(
        args.patterns
    )

    patterns = patterns_data["patterns"]

    pattern_count = len(
        patterns
    )

    print(
        f"Patterns: {pattern_count}"
    )

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    result, history = optimize(
        train=train,
        patterns_filename=args.patterns,
        pattern_count=pattern_count,
        seed=args.seed
    )

    optimized_weights = np.asarray(
        result.x,
        dtype=float
    )

    # --------------------------------------------------------
    # Print optimized weights
    # --------------------------------------------------------

    print_optimized_weights(
        args.patterns,
        optimized_weights
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print(
        "Evaluating optimized weights..."
    )

    train_iou, _ = evaluate_weights(
        train,
        optimized_weights
    )

    validation_iou, _ = evaluate_weights(
        validation,
        optimized_weights
    )

    test_iou, _ = evaluate_weights(
        test,
        optimized_weights
    )

    print()
    print("=" * 70)
    print("Final evaluation")
    print("=" * 70)

    print(
        f"Train IoU:      {train_iou:.6f}"
    )

    print(
        f"Validation IoU: {validation_iou:.6f}"
    )

    print(
        f"Test IoU:       {test_iou:.6f}"
    )

    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Save optimized patterns
    # --------------------------------------------------------

    save_optimized_patterns(
        args.patterns,
        optimized_weights,
        args.output_patterns
    )

    print(
        f"Optimized patterns saved to: "
        f"{args.output_patterns}"
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    save_history(
        history,
        args.history
    )

    print(
        f"Optimization history saved to: "
        f"{args.history}"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_iou_history(
        history,
        args.plot
    )

    print(
        f"IoU plot saved to: "
        f"{args.plot}"
    )


if __name__ == "__main__":
    main()