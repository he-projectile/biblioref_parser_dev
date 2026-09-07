import argparse
import json
import random
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution


REFERENCE_LABEL = "БИБЛ. ССЫЛКА"
DEFAULT_SEED = 42


# ============================================================
# Работа с разметкой
# ============================================================

def get_references(annotations):
    """Рекурсивно находит все библиографические ссылки."""

    result = []

    def walk(nodes):
        for node in nodes:
            if not isinstance(node, dict):
                continue

            if node.get("label") == REFERENCE_LABEL:
                result.append(node)

            children = node.get("children", [])
            if children:
                walk(children)

    walk(annotations)

    return result


def get_document_reference_lines(json_file, text_file):
    """Возвращает номера строк, входящих в библиографический блок."""

    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    references = get_references(
        data.get("annotations", [])
    )

    if not references:
        return set()

    line_starts = [0]

    for i, char in enumerate(text):
        if char == "\n":
            line_starts.append(i + 1)

    def char_to_line(position):
        left = 0
        right = len(line_starts) - 1

        while left <= right:
            mid = (left + right) // 2

            if line_starts[mid] <= position:
                left = mid + 1
            else:
                right = mid - 1

        return right + 1

    target_lines = set()

    for reference in references:
        start = reference.get("start")
        end = reference.get("end")

        if start is None or end is None:
            continue

        start_line = char_to_line(start)
        end_line = char_to_line(max(start, end - 1))

        for line in range(start_line, end_line + 1):
            target_lines.add(line)

    return target_lines


# ============================================================
# MACHINE JSON
# ============================================================

def load_machine_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Score
# ============================================================

def scores_from_counts(counts, weights):
    return counts @ weights


# ============================================================
# Поиск библиографического блока
# ============================================================

def find_best_block(scores, threshold):
    """
    Находит непрерывный блок строк, score которых >= threshold.

    Если блоков несколько, выбирается блок с максимальной
    суммой score.
    """

    best_start = None
    best_end = None
    best_value = -float("inf")

    current_start = None
    current_value = 0.0

    for i, score in enumerate(scores):

        if score >= threshold:

            if current_start is None:
                current_start = i
                current_value = score
            else:
                current_value += score

        else:

            if current_start is not None:

                if current_value > best_value:
                    best_value = current_value
                    best_start = current_start
                    best_end = i - 1

                current_start = None
                current_value = 0.0

    if current_start is not None:

        if current_value > best_value:
            best_value = current_value
            best_start = current_start
            best_end = len(scores) - 1

    if best_start is None:
        return None

    return best_start + 1, best_end + 1


# ============================================================
# IoU
# ============================================================

def interval_iou(predicted, target):
    if predicted is None or not target:
        return 0.0

    pred_start, pred_end = predicted

    target_start = min(target)
    target_end = max(target)

    intersection_start = max(
        pred_start,
        target_start
    )

    intersection_end = min(
        pred_end,
        target_end
    )

    if intersection_start <= intersection_end:
        intersection = (
            intersection_end
            - intersection_start
            + 1
        )
    else:
        intersection = 0

    union_start = min(
        pred_start,
        target_start
    )

    union_end = max(
        pred_end,
        target_end
    )

    union = union_end - union_start + 1

    if union == 0:
        return 0.0

    return intersection / union


# ============================================================
# Margin
# ============================================================

def calculate_margin(document, scores):
    """
    Разница между средним score библиографических
    и обычных строк.

    Используется только как слабый дополнительный
    критерий при оптимизации.
    """

    target = document["target"]

    positive_scores = []
    negative_scores = []

    for i, score in enumerate(scores, start=1):

        if i in target:
            positive_scores.append(score)
        else:
            negative_scores.append(score)

    if not positive_scores or not negative_scores:
        return 0.0

    return (
        np.mean(positive_scores)
        - np.mean(negative_scores)
    )


# ============================================================
# Dataset
# ============================================================

def prepare_dataset(source_dir):

    source_dir = Path(source_dir)

    dataset = []

    for text_file in sorted(source_dir.glob("*.txt")):

        if text_file.name.startswith("RECOGNISE_"):
            continue

        if text_file.name.startswith("MACHINE_"):
            continue

        json_file = text_file.with_suffix(".json")

        if not json_file.exists():
            continue

        machine_file = (
            source_dir /
            f"MACHINE_{text_file.stem}.json"
        )

        if not machine_file.exists():
            print(
                f"WARNING: no machine file for "
                f"{text_file.name}"
            )
            continue

        target_lines = get_document_reference_lines(
            json_file,
            text_file
        )

        if not target_lines:
            print(
                f"WARNING: no references in "
                f"{text_file.name}"
            )
            continue

        machine = load_machine_file(
            machine_file
        )

        counts = np.asarray(
            [
                line["counts"]
                for line in machine["lines"]
            ],
            dtype=float
        )

        dataset.append({
            "name": text_file.stem,
            "counts": counts,
            "target": target_lines
        })

    return dataset


# ============================================================
# Оценка одного документа
# ============================================================

def evaluate_document(
    document,
    weights,
    threshold
):
    scores = scores_from_counts(
        document["counts"],
        weights
    )

    predicted = find_best_block(
        scores,
        threshold
    )

    iou = interval_iou(
        predicted,
        document["target"]
    )

    margin = calculate_margin(
        document,
        scores
    )

    return iou, margin


# ============================================================
# Оценка dataset
# ============================================================

def evaluate_dataset(
    dataset,
    weights,
    threshold
):
    if not dataset:
        return 0.0, 0.0

    ious = []
    margins = []

    for document in dataset:

        iou, margin = evaluate_document(
            document,
            weights,
            threshold
        )

        ious.append(iou)
        margins.append(margin)

    return (
        float(np.mean(ious)),
        float(np.mean(margins))
    )


# ============================================================
# Optimization
# ============================================================

def optimize(train, pattern_count):

    dimension = pattern_count + 1

    weight_bounds = [
        (-10.0, 10.0)
        for _ in range(pattern_count)
    ]

    threshold_bounds = (0.0, 50.0)

    bounds = weight_bounds + [
        threshold_bounds
    ]

    # Очень маленький вклад margin.
    #
    # IoU остаётся главным критерием.
    #
    # Margin нужен только для того, чтобы отличать
    # решения с одинаковым IoU.

    MARGIN_COEFFICIENT = 0.001

    history = []

    def objective(parameters):

        weights = parameters[:-1]
        threshold = parameters[-1]

        mean_iou, mean_margin = evaluate_dataset(
            train,
            weights,
            threshold
        )

        # Максимизируем:
        #
        # IoU + lambda * margin
        #
        # scipy минимизирует функцию,
        # поэтому возвращаем отрицательное значение.

        objective_value = (
            mean_iou
            + MARGIN_COEFFICIENT * mean_margin
        )

        return -objective_value

    def callback(xk, convergence):

        weights = xk[:-1]
        threshold = xk[-1]

        mean_iou, mean_margin = evaluate_dataset(
            train,
            weights,
            threshold
        )

        history.append({
            "iteration": len(history) + 1,
            "best_iou": mean_iou,
            "mean_margin": mean_margin,
            "convergence": float(convergence),
        })

        print(
            f"Iteration "
            f"{len(history):3d} | "
            f"IoU = {mean_iou:.10f} | "
            f"Margin = {mean_margin:.6f}"
        )

        return False

    print()
    print("=" * 60)
    print(" OPTIMIZATION")
    print("=" * 60)
    print()

    print(f"Documents : {len(train)}")
    print(f"Patterns  : {pattern_count}")
    print(f"Parameters: {dimension}")
    print()

    result = differential_evolution(
        objective,
        bounds,
        seed=DEFAULT_SEED,

        # Размер популяции.
        # 5 * 430 ≈ 2150 кандидатов.
        popsize=5,

        maxiter=100,

        # Пока не делаем агрессивную остановку.
        # Хотим увидеть реальную динамику.
        tol=1e-7,

        polish=False,

        workers=1,

        updating="immediate",

        disp=False,

        callback=callback
    )

    weights = result.x[:-1]
    threshold = result.x[-1]

    return (
        weights,
        threshold,
        result,
        history
    )


# ============================================================
# Сохранение весов
# ============================================================

def save_weights(patterns_file, weights):

    with open(patterns_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    patterns = data.get("patterns", [])

    if len(patterns) != len(weights):
        raise RuntimeError(
            "Number of patterns changed!"
        )

    for pattern, weight in zip(
        patterns,
        weights
    ):
        pattern["weight"] = float(weight)

    data["patterns"] = patterns

    with open(patterns_file, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# Сохранение истории
# ============================================================

def save_history(
    source_dir,
    history
):

    source_dir = Path(source_dir)

    filename = (
        source_dir /
        "OPTIMIZATION_HISTORY.json"
    )

    data = {
        "iterations": history
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    return filename


# ============================================================
# Dataset result
# ============================================================

def print_dataset_result(
    name,
    dataset,
    weights,
    threshold
):

    iou, margin = evaluate_dataset(
        dataset,
        weights,
        threshold
    )

    print(
        f"{name:12s}: "
        f"IoU = {iou:.4f} "
        f"({len(dataset)} documents)"
    )

    print(
        f"{'':12s}  "
        f"Margin = {margin:.6f}"
    )

    return iou


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Optimize bibliographic pattern weights"
    )

    parser.add_argument(
        "source",
        help="Directory containing TXT and JSON dataset"
    )

    parser.add_argument(
        "-p",
        "--patterns",
        required=True,
        help="patterns.json"
    )

    parser.add_argument(
        "--train",
        type=float,
        default=0.6
    )

    parser.add_argument(
        "--validation",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(" LOADING DATASET")
    print("=" * 60)
    print()

    dataset = prepare_dataset(
        args.source
    )

    if len(dataset) < 3:
        raise RuntimeError(
            "Need at least 3 documents."
        )

    print(
        f"Documents found: {len(dataset)}"
    )

    random.shuffle(dataset)

    n = len(dataset)

    train_end = int(
        n * args.train
    )

    validation_end = (
        train_end
        + int(n * args.validation)
    )

    train = dataset[:train_end]

    validation = dataset[
        train_end:validation_end
    ]

    test = dataset[
        validation_end:
    ]

    print()
    print("Dataset split:")
    print(
        f"  Train      : {len(train)}"
    )
    print(
        f"  Validation : {len(validation)}"
    )
    print(
        f"  Test       : {len(test)}"
    )

    # --------------------------------------------------------
    # Patterns
    # --------------------------------------------------------

    with open(
        args.patterns,
        "r",
        encoding="utf-8"
    ) as f:
        pattern_data = json.load(f)

    patterns = pattern_data.get(
        "patterns",
        []
    )

    if not patterns:
        raise RuntimeError(
            "patterns.json contains no patterns."
        )

    pattern_count = len(patterns)

    print()
    print(
        f"Patterns: {pattern_count}"
    )

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    (
        weights,
        threshold,
        result,
        history
    ) = optimize(
        train,
        pattern_count
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(" RESULTS")
    print("=" * 60)
    print()

    print(
        f"Threshold: {threshold:.10f}"
    )

    print()

    print_dataset_result(
        "TRAIN",
        train,
        weights,
        threshold
    )

    print_dataset_result(
        "VALIDATION",
        validation,
        weights,
        threshold
    )

    print_dataset_result(
        "TEST",
        test,
        weights,
        threshold
    )

    # --------------------------------------------------------
    # Weights
    # --------------------------------------------------------

    print()
    print("Weights:")

    for i, (pattern, weight) in enumerate(
        zip(patterns, weights)
    ):

        print(
            f"  {i:3d} "
            f"{pattern.get('name', ''):35s} "
            f"{weight:12.6f}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_weights(
        args.patterns,
        weights
    )

    history_file = save_history(
        args.source,
        history
    )

    print()
    print("=" * 60)
    print(
        f"Weights written to: "
        f"{args.patterns}"
    )

    print(
        f"History written to: "
        f"{history_file}"
    )

    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
