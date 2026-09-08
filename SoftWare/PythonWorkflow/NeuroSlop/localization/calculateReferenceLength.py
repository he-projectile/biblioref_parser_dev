import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def process_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = get_reference_annotations(data)

    lengths = []

    for start, end in annotations:
        length = end - start + 1
        lengths.append(length)

    return lengths


def main():

    parser = argparse.ArgumentParser(
        description="Calculate lengths of bibliographic references"
    )

    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing annotation JSON files"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Histogram output filename"
    )

    args = parser.parse_args()

    directory = args.directory

    # MACHINE_*.json исключаем
    json_files = sorted(
        f for f in directory.glob("*.json")
        if not f.name.startswith("MACHINE_")
    )

    print(f"Found {len(json_files)} annotation files.")

    all_lengths = []

    for filename in json_files:

        try:
            lengths = process_file(filename)

            all_lengths.extend(lengths)

            print(
                f"{filename.name}: "
                f"{len(lengths)} references"
            )

        except Exception as e:
            print(
                f"ERROR: {filename.name}: {e}"
            )

    if not all_lengths:
        print("No bibliographic references found.")
        return

    # Максимальная длина ссылки
    max_length = max(all_lengths)

    # length_counts[i] = количество ссылок длины i
    length_counts = [0] * (max_length + 1)

    for length in all_lengths:
        length_counts[length] += 1

    print()
    print("=" * 70)
    print("REFERENCE LENGTH DISTRIBUTION")
    print("=" * 70)

    print(f"Documents          : {len(json_files)}")
    print(f"References         : {len(all_lengths)}")
    print(f"Min length         : {min(all_lengths)}")
    print(f"Max length         : {max(all_lengths)}")
    print(f"Mean length        : {np.mean(all_lengths):.2f}")
    print(f"Median length      : {np.median(all_lengths):.2f}")

    print()
    print("ARRAY:")
    print(length_counts)

    # Гистограмма
    plt.figure(
        figsize=(12, 6),
        dpi=150
    )

    plt.hist(
        all_lengths,
        bins=np.arange(
            min(all_lengths),
            max(all_lengths) + 2
        ) - 0.5,
        edgecolor="black"
    )

    plt.xlabel("Длина библиографической ссылки, символов")
    plt.ylabel("Количество ссылок")

    plt.title(
        "Распределение длин библиографических ссылок"
    )

    plt.grid(
        True,
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    if args.output is None:
        output_filename = (
            directory / "reference_lengths_histogram.png"
        )
    else:
        output_filename = args.output

    plt.savefig(
        output_filename,
        dpi=300
    )

    plt.close()

    print()
    print(f"Histogram: {output_filename}")


if __name__ == "__main__":
    main()