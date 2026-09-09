# Пути к файлам и папкам
$SCRIPT_PATH = ".\NeuroSlop\localization\patternRecognition.py"
$PATTERNS_PATH = ".\NeuroSlop\localization\patterns_optimized.json"
$OUTPUT_DIR = ".\NeuroSlop\localization\localizationProducts"
$INPUT_DIR = ".\DataSource"

# Создаем папку для результатов, если её нет
if (-not (Test-Path $OUTPUT_DIR)) {
    New-Item -ItemType Directory -Path $OUTPUT_DIR | Out-Null
}

# Ищем все файлы .txt в папке DataSource
$files = Get-ChildItem -Path "$INPUT_DIR\*.txt" -ErrorAction SilentlyContinue

if ($files.Count -eq 0) {
    Write-Host "txt files not found in $INPUT_DIR" -ForegroundColor Yellow
    Exit
}

# Цикл обработки каждого файла
foreach ($file in $files) {
    Write-Host "Processing file: $($file.FullName)" -ForegroundColor Cyan
    
    # Запуск Python скрипта
    python $SCRIPT_PATH $file.FullName --patterns $PATTERNS_PATH --output-dir $OUTPUT_DIR
}

Write-Host "All files processed successfully!" -ForegroundColor Green
