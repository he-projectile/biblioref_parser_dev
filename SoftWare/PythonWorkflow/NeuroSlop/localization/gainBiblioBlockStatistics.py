import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def get_biblio_bounds(annotation_data):
    references = []

    for annotation in annotation_data.get("annotations", []):
        if annotation.get("label") == "БИБЛ. ССЫЛКА":
            references.append(annotation)

    if not references:
        return None, None

    start = min(
        annotation["start"]
        for annotation in references
    )

    end = max(
        annotation["end"]
        for annotation in references
    )

    return start, end


def get_biblio_line_bounds(text, start, end):
    """
    Переводит символьные координаты блока
    в координаты строк.

    Возвращает:
        block_start_line  - номер первой строки блока
        block_end_line    - номер последней строки блока
        block_line_count  - количество строк блока
    """

    offset = 0
    start_line = None
    end_line = None

    for line_number, line in enumerate(
        text.splitlines(keepends=True),
        start=1
    ):
        line_start = offset
        line_end = offset + len(line)

        # Строка пересекается с блоком
        if line_end > start and line_start < end:

            if start_line is None:
                start_line = line_number

            end_line = line_number

        offset = line_end

    if start_line is None:
        return None, None, 0

    return (
        start_line,
        end_line,
        end_line - start_line + 1
    )


def get_biblio_line_lengths(text, start, end):
    line_lengths = []

    offset = 0

    for line in text.splitlines(keepends=True):
        line_start = offset
        line_end = offset + len(line)

        # Строка пересекается с библиографическим блоком
        if line_end > start and line_start < end:
            line_without_newline = line.rstrip("\r\n")

            line_lengths.append(
                len(line_without_newline)
            )

        offset = line_end

    return line_lengths


def collect_data(data_dir):
    # Для первых двух графиков теперь используем строки
    document_lengths = []
    block_lengths = []
    block_starts = []

    # Оставляем статистику длин отдельных строк
    all_line_lengths = []

    processed = 0
    skipped = 0

    txt_files = sorted(data_dir.glob("*.txt"))

    print(
        f"Найдено TXT-файлов: {len(txt_files)}"
    )
    print()

    for txt_file in txt_files:

        json_file = txt_file.with_suffix(".json")

        if not json_file.exists():
            print(
                f"[SKIP] Нет JSON для: "
                f"{txt_file.name}"
            )
            skipped += 1
            continue

        try:
            text = txt_file.read_text(
                encoding="utf-8"
            )

            annotation_data = json.loads(
                json_file.read_text(
                    encoding="utf-8"
                )
            )

        except Exception as error:
            print(
                f"[ERROR] {txt_file.name}: {error}"
            )
            skipped += 1
            continue

        block_start, block_end = get_biblio_bounds(
            annotation_data
        )

        if block_start is None:
            print(
                f"[SKIP] Нет БИБЛ. ССЫЛКА: "
                f"{txt_file.name}"
            )
            skipped += 1
            continue

        # -----------------------------------------
        # Координаты библиографического блока
        # в строках
        # -----------------------------------------

        (
            block_start_line,
            block_end_line,
            block_line_count
        ) = get_biblio_line_bounds(
            text,
            block_start,
            block_end
        )

        if block_start_line is None:
            print(
                f"[SKIP] Не удалось определить "
                f"строки блока: {txt_file.name}"
            )
            skipped += 1
            continue

        # Количество строк во всём документе
        document_line_count = len(
            text.splitlines()
        )

        # Длины отдельных строк блока
        line_lengths = get_biblio_line_lengths(
            text,
            block_start,
            block_end
        )

        document_lengths.append(
            document_line_count
        )

        block_lengths.append(
            block_line_count
        )

        block_starts.append(
            block_start_line
        )

        all_line_lengths.extend(
            line_lengths
        )

        processed += 1

        print(
            f"[OK] {txt_file.name}: "
            f"document_lines={document_line_count}, "
            f"block_start_line={block_start_line}, "
            f"block_end_line={block_end_line}, "
            f"block_lines={block_line_count}, "
            f"block_line_lengths={len(line_lengths)}"
        )

    print()
    print(
        f"Обработано документов: {processed}"
    )

    print(
        f"Пропущено документов:   {skipped}"
    )

    print(
        f"Всего строк в библиоблоках: "
        f"{len(all_line_lengths)}"
    )

    return (
        np.array(
            document_lengths,
            dtype=float
        ),
        np.array(
            block_lengths,
            dtype=float
        ),
        np.array(
            block_starts,
            dtype=float
        ),
        all_line_lengths
    )


def calculate_linear_approximation(x, y):
    """
    Линейная аппроксимация:

        y = k * x + b

    Возвращает:
        k
        b
        r2
    """

    if len(x) < 2:
        return None, None, None

    k, b = np.polyfit(x, y, 1)

    y_pred = k * x + b

    ss_res = np.sum(
        (y - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y - np.mean(y)) ** 2
    )

    if ss_tot == 0:
        r2 = 1.0
    else:
        r2 = 1.0 - ss_res / ss_tot

    return k, b, r2


def plot_block_length(
    document_lengths,
    block_lengths,
    output_file
):
    k, b, r2 = calculate_linear_approximation(
        document_lengths,
        block_lengths
    )
    residuals = block_lengths - (k * document_lengths + b)
    std = np.std(residuals)

    plt.figure(figsize=(10, 6))

    plt.scatter(
        document_lengths,
        block_lengths,
        alpha=0.7,
        label="Данные"
    )

    if k is not None:
        x_line = np.linspace(
            document_lengths.min(),
            document_lengths.max(),
            200
        )

        y_line = (
            k * x_line
            + b
        )

        plt.plot(
            x_line,
            y_line,
            linewidth=2,
            label="Линейная аппроксимация"
        )

        equation = (
            f"y = {k:.6g}x "
            f"{'+' if b >= 0 else '-'} "
            f"{abs(b):.6g}\n"
            f"R² = {r2:.4f}\n"
            f"СКО = {std:.2f} строк"
        )

        plt.text(
            0.02,
            0.98,
            equation,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round",
                alpha=0.8
            )
        )

    plt.xlabel(
        "Количество строк в документе"
    )

    plt.ylabel(
        "Длина библиографического блока, строк"
    )

    plt.title(
        "Зависимость длины библиографического блока "
        "от длины документа"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=150
    )

    plt.close()

    print(
        f"Сохранён график: {output_file}"
    )

    if k is not None:
        print(
            f"  Аппроксимация: "
            f"y = {k:.6g}x + {b:.6g}"
        )

        print(
            f"  R² = {r2:.6f}"
        )

        print(
            f"  СКО = {std:.6f} строк"
        )


def plot_block_start(
    document_lengths,
    block_starts,
    output_file
):
    k, b, r2 = calculate_linear_approximation(
        document_lengths,
        block_starts
    )
    residuals = block_starts - (k * document_lengths + b)
    std = np.std(residuals)

    plt.figure(figsize=(10, 6))

    plt.scatter(
        document_lengths,
        block_starts,
        alpha=0.7,
        label="Данные"
    )

    if k is not None:
        x_line = np.linspace(
            document_lengths.min(),
            document_lengths.max(),
            200
        )

        y_line = (
            k * x_line
            + b
        )

        plt.plot(
            x_line,
            y_line,
            linewidth=2,
            label="Линейная аппроксимация"
        )

        equation = (
            f"y = {k:.6g}x "
            f"{'+' if b >= 0 else '-'} "
            f"{abs(b):.6g}\n"
            f"R² = {r2:.4f}\n"
            f"СКО = {std:.2f} строк"
        )

        plt.text(
            0.02,
            0.98,
            equation,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round",
                alpha=0.8
            )
        )

    plt.xlabel(
        "Количество строк в документе"
    )

    plt.ylabel(
        "Начало библиографического блока, строка"
    )

    plt.title(
        "Зависимость положения начала "
        "библиографического блока "
        "от длины документа"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=150
    )

    plt.close()

    print(
        f"Сохранён график: {output_file}"
    )

    if k is not None:
        print(
            f"  Аппроксимация: "
            f"y = {k:.6g}x + {b:.6g}"
        )

        print(
            f"  R² = {r2:.6f}"
        )

        print(
            f"  СКО = {std:.6f} строк"
        )


def plot_line_lengths(
    line_lengths,
    output_file
):
    if not line_lengths:
        print(
            "Нет данных о длинах строк."
        )
        return

    max_length = max(
        line_lengths
    )

    counts = np.bincount(
        line_lengths,
        minlength=max_length + 1
    )

    x = np.arange(
        len(counts)
    )

    mask = counts > 0

    plt.figure(figsize=(12, 6))

    plt.scatter(
        x[mask],
        counts[mask],
        s=20
    )

    plt.xlabel(
        "Длина строки, символы"
    )

    plt.ylabel(
        "Количество строк"
    )

    plt.title(
        "Распределение длин строк "
        "в библиографических блоках"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=150
    )

    plt.close()

    print(
        f"Сохранён график: {output_file}"
    )

    return counts


def save_line_length_array(
    line_lengths,
    output_file
):
    if not line_lengths:
        counts = np.array(
            [0],
            dtype=int
        )
    else:
        max_length = max(
            line_lengths
        )

        counts = np.bincount(
            line_lengths,
            minlength=max_length + 1
        )

    array_string = (
        "["
        + ",".join(
            str(int(value))
            for value in counts
        )
        + "]"
    )

    output_file.write_text(
        array_string,
        encoding="utf-8"
    )

    print(
        f"Массив длин строк сохранён: "
        f"{output_file}"
    )

    print(
        f"Максимальная длина строки: "
        f"{len(counts) - 1}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Анализ библиографических блоков "
            "по TXT/JSON-файлам"
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=(
            "Директория с TXT и JSON файлами"
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Директория для сохранения "
            "графиков и массива"
        )
    )

    args = parser.parse_args()

    data_dir = (
        args.data_dir.resolve()
    )

    output_dir = (
        args.output_dir.resolve()
    )

    if not data_dir.exists():
        print(
            f"Ошибка: директория не существует:\n"
            f"{data_dir}"
        )
        return

    if not data_dir.is_dir():
        print(
            f"Ошибка: это не директория:\n"
            f"{data_dir}"
        )
        return

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Директория данных:\n"
        f"  {data_dir}"
    )

    print(
        f"Директория результатов:\n"
        f"  {output_dir}"
    )

    print()

    (
        document_lengths,
        block_lengths,
        block_starts,
        line_lengths
    ) = collect_data(
        data_dir
    )

    if len(document_lengths) == 0:
        print(
            "Нет данных для построения графиков."
        )
        return

    plot_block_length(
        document_lengths,
        block_lengths,
        output_dir
        / "biblio_block_length_vs_document.png"
    )

    plot_block_start(
        document_lengths,
        block_starts,
        output_dir
        / "biblio_block_start_vs_document.png"
    )

    plot_line_lengths(
        line_lengths,
        output_dir
        / "biblio_line_lengths_scatter.png"
    )

    save_line_length_array(
        line_lengths,
        output_dir
        / "biblio_line_lengths.txt"
    )

    print()
    print("Готово.")


if __name__ == "__main__":
    main()
