#!/bin/bash

# Путь к питон-скрипту
SCRIPT_PATH="./NeuroSlop/localization/patternRecognition.py"

# Путь к файлу паттернов
PATTERNS_PATH="./NeuroSlop/localization/patterns.json"

# Директория для сохранения результатов
OUTPUT_DIR="./NeuroSlop/localization/localizationProducts"

# Папка, в которой лежат исходные .txt файлы
INPUT_DIR="./DataSource"

# Проверяем, существует ли папка с результатами, если нет — создаем
mkdir -p "$OUTPUT_DIR"

# Перебираем все файлы .txt в указанной папке
for file in "$INPUT_DIR"/*.txt; do
    # Проверяем, что файл действительно существует (на случай, если папка пуста)
    [ -e "$file" ] || continue
    
    echo "Обработка файла: $file"
    
    # Запуск питон-скрипта с передачей текущего файла
    python "$SCRIPT_PATH" "$file" --patterns "$PATTERNS_PATH" --output-dir "$OUTPUT_DIR"
done

echo "Все файлы успешно обработаны!"
