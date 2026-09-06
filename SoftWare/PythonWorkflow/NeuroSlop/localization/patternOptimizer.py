import argparse
import json
import random
import shutil
import subprocess
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
    """
    Возвращает множество строк, которые относятся
    к библиографическому блоку.

    Индексы start/end в разметке считаются по символам текста.
    """

    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data.get("annotations", [])

    references = get_references(annotations)

    if not references:
        return set()

    # Позиция начала каждой строки
    line_starts = [0]

    for i, char in enumerate(text):
        if char == "\n":
            line_starts.append(i + 1)

    def char_to_line(position):
        """
        Перевод позиции символа в номер строки.
        """

        # binary search
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
# Формирование предсказанного блока
# ============================================================

def scores_from_counts(counts, weights):
    """
    S_i = sum_j x_ij * w_j
    """

    return np.asarray(counts) @ np.asarray(weights)


def find_best_block(scores, threshold):
    """
    Находит непрерывный блок строк, где score >= threshold.

    Из нескольких блоков выбирается блок с максимальной
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

    # Последний блок
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
    """
    IoU двух интервалов строк.
    """

    if predicted is None or not target:
        return 0.0

    pred_start, pred_end = predicted

    target_start = min(target)
    target_end = max(target)

    intersection_start = max(pred_start, target_start)
    intersection_end = min(pred_end, target_end)

    if intersection_start > intersection_end:
        intersection = 0
    else:
        intersection = intersection_end - intersection_start + 1

    union_start = min(pred_start, target_start)
    union_end = max(pred_end, target_end)

    union = union_end - union_start + 1

    if union == 0:
        return 0.0

    return intersection / union


# ============================================================
# Dataset
# ============================================================

def prepare_dataset(source_dir):
    """
    Находит пары:
        document.txt
        document.json

    и соответствующий:
        MACHINE_document.json
    """

    source_dir = Path(source_dir)

    dataset = []

    for text_file in sorted(source_dir.glob("*.txt")):

        # Не брать наши результаты
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

        machine = load_machine_file(machine_file)

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
            "target": target_lines,
        })

    return dataset


# ============================================================
# Метрики
# ============================================================

def evaluate_document(document, weights, threshold):
    scores = scores_from_counts(
        document["counts"],
        weights
    )

    predicted = find_best_block(
        scores,
        threshold
    )

    return interval_iou(
        predicted,
        document["target"]
    )


def evaluate_dataset(dataset, weights, threshold):
    if not dataset:
        return 0.0

    values = [
        evaluate_document(
            document,
            weights,
            threshold
        )
        for document in dataset
    ]

    return float(np.mean(values))


# ============================================================
# Оптимизация
# ============================================================

def optimize(train, pattern_count):
    """
    Оптимизирует:

        weight_1 ... weight_N
        threshold

    Целевая функция:

        mean(IoU)
    """

    # Первые N параметров — веса.
    # Последний — threshold.

    dimension = pattern_count + 1

    # Ограничения весов.
    #
    # Разрешаем отрицательные веса:
    #
    # отрицательный вес означает, что наличие паттерна
    # скорее говорит ПРОТИВ библиографической строки.
    #
    weight_bounds = [
        (-10.0, 10.0)
        for _ in range(pattern_count)
    ]

    # Порог.
    threshold_bounds = (0.0, 50.0)

    bounds = weight_bounds + [threshold_bounds]

    def objective(parameters):

        weights = parameters[:-1]
        threshold = parameters[-1]

        score = evaluate_dataset(
            train,
            weights,
            threshold
        )

        return -score

    print()
    print("=" * 60)
    print(" OPTIMIZATION")
    print("=" * 60)
    print()

    print(f"Documents : {len(train)}")
    print(f"Patterns  : {pattern_count}")
    print()

    result = differential_evolution(
        objective,
        bounds,
        seed=DEFAULT_SEED,
        popsize=5,
        maxiter=100,
        tol=1e-5,
        polish=False,
        workers=1,
        updating="immediate",
        disp=True
    )

    weights = result.x[:-1]
    threshold = result.x[-1]

    return weights, threshold, result


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

    for pattern, weight in zip(patterns, weights):
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
# Печать результатов
# ============================================================

def print_dataset_result(name, dataset, weights, threshold):

    score = evaluate_dataset(
        dataset,
        weights,
        threshold
    )

    print(
        f"{name:12s}: "
        f"IoU = {score:.4f} "
        f"({len(dataset)} documents)"
    )

    return score


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
        default=0.6,
        help="Train fraction"
    )

    parser.add_argument(
        "--validation",
        type=float,
        default=0.2,
        help="Validation fraction"
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

    dataset = prepare_dataset(args.source)

    if len(dataset) < 3:
        raise RuntimeError(
            "Need at least 3 documents."
        )

    print(f"Documents found: {len(dataset)}")

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    random.shuffle(dataset)

    n = len(dataset)

    train_end = int(
        n * args.train
    )

    validation_end = train_end + int(
        n * args.validation
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
    print(f"  Train      : {len(train)}")
    print(f"  Validation : {len(validation)}")
    print(f"  Test       : {len(test)}")

    # --------------------------------------------------------
    # Patterns
    # --------------------------------------------------------

    with open(args.patterns, "r", encoding="utf-8") as f:
        pattern_data = json.load(f)

    patterns = pattern_data.get("patterns", [])

    if not patterns:
        raise RuntimeError(
            "patterns.json contains no patterns."
        )

    pattern_count = len(patterns)

    print()
    print(f"Patterns: {pattern_count}")

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    weights, threshold, result = optimize(
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

    print(f"Threshold: {threshold:.6f}")
    print()

    print("Weights:")

    for i, (pattern, weight) in enumerate(
        zip(patterns, weights)
    ):
        print(
            f"  {i:3d} "
            f"{pattern.get('name', ''):30s} "
            f"{weight:10.6f}"
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
    # Сохраняем веса
    # --------------------------------------------------------

    save_weights(
        args.patterns,
        weights
    )

    print()
    print("=" * 60)
    print(f"Weights written to: {args.patterns}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()