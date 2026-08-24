from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

from loader import DatasetLoader
from blocks import group_references
from lines import split_lines, label_lines


def inspect_document(document):
    lines = split_lines(document.text)
    labeled_lines = label_lines(document, lines)
    blocks = group_references(document)

    positive_lines = sum(
        1
        for _, positive in labeled_lines
        if positive
    )

    negative_lines = len(labeled_lines) - positive_lines

    return {
        "lines": lines,
        "positive_lines": positive_lines,
        "negative_lines": negative_lines,
        "blocks": blocks,
    }


def main():
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        data_dir = (
            Path(__file__).resolve().parent.parent / "DataSource"
        )

    print("=" * 70)
    print("BIBLIOGRAPHY LOCALIZATION DATASET INSPECTION")
    print("=" * 70)

    print(f"\nDataSource: {data_dir}")

    loader = DatasetLoader(data_dir)

    documents = loader.load_all()

    if not documents:
        print("\n[ERROR] Документы не найдены.")
        return 1

    print(f"\nDocuments loaded: {len(documents)}")

    total_annotations = 0
    total_references = 0
    total_lines = 0
    total_positive_lines = 0
    total_negative_lines = 0

    documents_with_references = 0

    block_counts = []
    reference_counts = []
    line_counts = []

    print("\n" + "-" * 70)
    print("DOCUMENTS")
    print("-" * 70)

    for document in documents:
        references = document.references

        inspection = inspect_document(document)

        blocks = inspection["blocks"]

        total_annotations += len(document.annotations)
        total_references += len(references)

        total_lines += len(inspection["lines"])
        total_positive_lines += inspection["positive_lines"]
        total_negative_lines += inspection["negative_lines"]

        if references:
            documents_with_references += 1

        block_counts.append(len(blocks))
        reference_counts.append(len(references))
        line_counts.append(len(inspection["lines"]))

        print(
            f"{document.name[:45]:45} "
            f"refs={len(references):4d} "
            f"blocks={len(blocks):3d} "
            f"lines={len(inspection['lines']):6d}"
        )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nDocuments:")
    print(f"  total:                 {len(documents)}")
    print(
        f"  with references:       "
        f"{documents_with_references}"
    )
    print(
        f"  without references:    "
        f"{len(documents) - documents_with_references}"
    )

    print(f"\nAnnotations:")
    print(f"  total:                 {total_annotations}")
    print(f"  bibliography refs:     {total_references}")

    print(f"\nLines:")
    print(f"  total:                 {total_lines}")
    print(f"  positive:              {total_positive_lines}")
    print(f"  negative:              {total_negative_lines}")

    if total_lines:
        positive_ratio = (
            total_positive_lines / total_lines
        )

        print(
            f"  positive ratio:        "
            f"{positive_ratio:.4f}"
        )

    print(f"\nBlocks:")
    print(
        f"  total:                 "
        f"{sum(block_counts)}"
    )

    if reference_counts:
        print(
            f"  references/document:   "
            f"{sum(reference_counts) / len(reference_counts):.2f}"
        )

        print(
            f"  min references/doc:    "
            f"{min(reference_counts)}"
        )

        print(
            f"  max references/doc:    "
            f"{max(reference_counts)}"
        )

    if block_counts:
        print(
            f"  blocks/document:       "
            f"{sum(block_counts) / len(block_counts):.2f}"
        )

    if line_counts:
        print(
            f"\nDocument length in lines:"
        )

        print(
            f"  min:                   "
            f"{min(line_counts)}"
        )

        print(
            f"  max:                   "
            f"{max(line_counts)}"
        )

        print(
            f"  mean:                  "
            f"{sum(line_counts) / len(line_counts):.2f}"
        )

    print("\n" + "=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())