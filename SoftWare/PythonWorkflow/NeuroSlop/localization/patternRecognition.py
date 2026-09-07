import argparse
import json
import re
from pathlib import Path


def load_patterns(filename):
    """Загрузка паттернов и их весов из JSON."""

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    patterns = []

    for pattern in data.get("patterns", []):
        regex = pattern.get("regex")
        weight = pattern.get("weight", 0.0)

        if not regex:
            continue

        try:
            compiled = re.compile(regex, re.IGNORECASE)
        except re.error as e:
            print(
                f"WARNING: invalid regex "
                f"'{pattern.get('name', 'UNKNOWN')}': {e}"
            )
            continue

        patterns.append({
            "name": pattern.get("name", ""),
            "regex": regex,
            "weight": float(weight),
            "compiled": compiled
        })

    return patterns


def recognize_lines(lines, patterns):
    """
    Для каждой строки:
        x_ij = количество срабатываний j-го паттерна
        score = sum(x_ij * weight_j)
    """

    result = []

    for line_number, line in enumerate(lines, start=1):

        counts = []
        score = 0.0

        for pattern in patterns:
            count = len(pattern["compiled"].findall(line))

            counts.append(count)
            score += count * pattern["weight"]

        result.append({
            "line": line_number,
            "text": line,
            "counts": counts,
            "score": score
        })

    return result


def save_human_readable(filename, results, patterns):
    """
    Человекочитаемый результат.

    Формат:
        номер | score | текст
    """

    with open(filename, "w", encoding="utf-8") as f:

        f.write("PATTERN RECOGNITION\n")
        f.write("=" * 80 + "\n\n")

        f.write("Patterns:\n")

        for i, pattern in enumerate(patterns):
            f.write(
                f"  {i:3d}: "
                f"{pattern['name']} "
                f"(weight={pattern['weight']:.6f})\n"
            )

        f.write("\n")
        f.write("=" * 80 + "\n\n")

        f.write(
            f"{'LINE':>6} | "
            f"{'SCORE':>10} | "
            f"TEXT\n"
        )

        f.write("-" * 80 + "\n")

        for item in results:
            text = item["text"].replace("\t", " ")

            f.write(
                f"{item['line']:6d} | "
                f"{item['score']:10.4f} | "
                f"{text}\n"
            )

        scores = [item["score"] for item in results]

        if scores:
            min_score = min(scores)
            max_score = max(scores)
            mean_score = sum(scores) / len(scores)
        else:
            min_score = 0.0
            max_score = 0.0
            mean_score = 0.0

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("STATISTICS\n")
        f.write("=" * 80 + "\n")

        f.write(f"Lines : {len(results)}\n")
        f.write(f"Min   : {min_score:.4f}\n")
        f.write(f"Max   : {max_score:.4f}\n")
        f.write(f"Mean  : {mean_score:.4f}\n")


def save_machine_data(filename, results, patterns):
    """
    Машинный формат.

    ВАЖНО:
    score сюда не записывается, поскольку он зависит от весов.

    Сохраняются только:
        counts = X[i,j]

    Это позволяет потом перебирать веса без повторного
    запуска распознавания текста.
    """

    machine_data = {
        "patterns": [
            {
                "name": pattern["name"],
                "weight": pattern["weight"]
            }
            for pattern in patterns
        ],

        "lines": [
            {
                "line": item["line"],
                "counts": item["counts"]
            }
            for item in results
        ]
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            machine_data,
            f,
            ensure_ascii=False,
            indent=2
        )


def print_console_statistics(input_file, patterns, results):
    """Красивый вывод в консоль."""

    scores = [item["score"] for item in results]

    if scores:
        min_score = min(scores)
        max_score = max(scores)
        mean_score = sum(scores) / len(scores)
    else:
        min_score = 0.0
        max_score = 0.0
        mean_score = 0.0

    print()
    print("=" * 50)
    print(" Pattern Recognition")
    print("=" * 50)

    print()
    print(f"Input file       : {input_file}")
    print(f"Patterns loaded  : {len(patterns)}")
    print(f"Lines processed  : {len(results)}")

    print()
    print("Score statistics:")
    print(f"  Min            : {min_score:.4f}")
    print(f"  Max            : {max_score:.4f}")
    print(f"  Mean           : {mean_score:.4f}")

    print()
    print("Top scoring lines:")

    # Показываем несколько самых сильных строк,
    # чтобы в консоли было видно, что алгоритм вообще нашёл.
    top_lines = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )[:10]

    for item in top_lines:
        text = item["text"].strip()

        if len(text) > 100:
            text = text[:97] + "..."

        print(
            f"  {item['line']:5d} | "
            f"{item['score']:8.3f} | "
            f"{text}"
        )

    print()
    print("=" * 50)


def main():

    parser = argparse.ArgumentParser(
        description="Bibliographic pattern recognition"
    )

    parser.add_argument(
        "input",
        help="Input TXT file"
    )

    parser.add_argument(
        "-p",
        "--patterns",
        required=True,
        help="JSON file with mined patterns"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    patterns_path = Path(args.patterns)

    # ------------------------------------------------------------
    # Формирование имён выходных файлов
    # ------------------------------------------------------------

    human_output = (
        input_path.parent /
        f"RECOGNISE_{input_path.stem}.txt"
    )

    machine_output = (
        input_path.parent /
        f"MACHINE_{input_path.stem}.json"
    )

    # ------------------------------------------------------------
    # Загрузка текста
    # ------------------------------------------------------------

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    lines = text.splitlines()

    # ------------------------------------------------------------
    # Загрузка паттернов
    # ------------------------------------------------------------

    patterns = load_patterns(patterns_path)

    if not patterns:
        print("ERROR: no valid patterns found.")
        return 1

    # ------------------------------------------------------------
    # Распознавание
    # ------------------------------------------------------------

    results = recognize_lines(lines, patterns)

    # ------------------------------------------------------------
    # Вывод
    # ------------------------------------------------------------

    save_human_readable(
        human_output,
        results,
        patterns
    )

    save_machine_data(
        machine_output,
        results,
        patterns
    )

    # ------------------------------------------------------------
    # Консоль
    # ------------------------------------------------------------

    print_console_statistics(
        input_path,
        patterns,
        results
    )

    print()
    print(f"Human readable : {human_output}")
    print(f"Machine data   : {machine_output}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())