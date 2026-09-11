import argparse
import json
import re
from pathlib import Path


def load_patterns(filename):
    """
    Загружает паттерны из JSON-файла.

    Ожидаемый формат:
    {
        "patterns": [
            {
                "name": "...",
                "regex": "..."
            }
        ]
    }
    """

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    patterns = []

    for pattern in data["patterns"]:
        regex = pattern["regex"]

        try:
            compiled = re.compile(regex)
        except re.error as e:
            print(f"Ошибка компиляции regex '{regex}': {e}")
            continue

        patterns.append({
            "name": pattern["name"],
            "regex": regex,
            "compiled": compiled
        })

    return patterns


def recognize_lines(lines, patterns):
    """
    Для каждой строки считает количество срабатываний
    каждого паттерна.

    Возвращает список:
    {
        "line": номер строки,
        "length": длина строки,
        "counts": [количество совпадений каждого паттерна],
        "text": текст строки
    }
    """

    result = []

    for line_number, line in enumerate(lines, start=1):
        counts = []

        for pattern in patterns:
            matches = pattern["compiled"].findall(line)
            counts.append(len(matches))

        result.append({
            "line": line_number,
            "length": len(line),
            "counts": counts,
            "text": line.rstrip("\n")
        })

    return result


def save_human_readable(filename, lines_data, patterns):
    """
    Сохраняет человекочитаемый результат.
    """

    with open(filename, "w", encoding="utf-8") as f:

        f.write("PATTERN RECOGNITION RESULT\n")
        f.write("=" * 80 + "\n\n")

        f.write("PATTERNS:\n")
        for i, pattern in enumerate(patterns):
            f.write(f"{i}: {pattern['name']}\n")
            f.write(f"   regex: {pattern['regex']}\n")

        f.write("\n")
        f.write("=" * 80 + "\n\n")

        for item in lines_data:
            f.write(
                f"LINE {item['line']} "
                f"(length={item['length']})\n"
            )

            f.write(f"TEXT: {item['text']}\n")

            f.write("COUNTS:\n")

            for i, count in enumerate(item["counts"]):
                if count > 0:
                    f.write(
                        f"  {patterns[i]['name']}: {count}\n"
                    )

            f.write("\n")


def save_machine_data(filename, lines_data, patterns):
    """
    Сохраняет машинно-обрабатываемый JSON.
    """

    data = {
        "patterns": [
            {
                "name": pattern["name"],
                "regex": pattern["regex"]
            }
            for pattern in patterns
        ],
        "lines": lines_data
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


def main():
    parser = argparse.ArgumentParser(
        description="Recognize bibliographic patterns in text"
    )

    parser.add_argument(
        "input",
        help="Input TXT file"
    )

    parser.add_argument(
        "--patterns",
        required=True,
        help="JSON file with regex patterns"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for MACHINE and HUMAN output files"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    patterns_path = Path(args.patterns)
    output_dir = Path(args.output_dir)

    # Проверяем входные файлы
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    if not patterns_path.exists():
        raise FileNotFoundError(
            f"Patterns file not found: {patterns_path}"
        )

    # Создаём директорию результатов
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Загружаем текст
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Загружаем паттерны
    patterns = load_patterns(patterns_path)

    print(f"Input:       {input_path}")
    print(f"Patterns:    {patterns_path}")
    print(f"Output dir:  {output_dir}")
    print(f"Lines:       {len(lines)}")
    print(f"Patterns:    {len(patterns)}")

    # Распознавание
    lines_data = recognize_lines(
        lines,
        patterns
    )

    # Формируем имена выходных файлов
    stem = input_path.stem

    machine_filename = output_dir / f"MACHINE_{stem}.json"
    human_filename = output_dir / f"HUMAN_{stem}.txt"

    # Сохраняем результаты
    save_machine_data(
        machine_filename,
        lines_data,
        patterns
    )

    save_human_readable(
        human_filename,
        lines_data,
        patterns
    )

    print()
    print("Done.")
    print(f"MACHINE: {machine_filename}")
    print(f"HUMAN:   {human_filename}")


if __name__ == "__main__":
    main()