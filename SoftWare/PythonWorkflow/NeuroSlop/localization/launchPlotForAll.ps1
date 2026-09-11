# Пути к файлам и папкам
$SCRIPT_PATH = ".\NeuroSlop\localization\plotBiblioBlockLocalization.py"
$PATTERNS_PATH = ".\NeuroSlop\localization\patterns_optimized_9.json"
$OUTPUT_DIR = ".\NeuroSlop\localization\localizationProducts"
$INPUT_DIR = ".\DataSource"

# Ищем все исходные файлы .json в папке DataSource
$files = Get-ChildItem -Path "$INPUT_DIR\*.json" -ErrorAction SilentlyContinue

if ($files.Count -eq 0) {
    Write-Host "JSON files not found in $INPUT_DIR" -ForegroundColor Yellow
    Exit
}

# Цикл обработки каждой пары файлов
foreach ($file in $files) {
    # Формируем имя файла с результатом (добавляем префикс MACHINE_)
    $machineFileName = "MACHINE_" + $file.Name
    $machineFilePath = Join-Path $OUTPUT_DIR $machineFileName

    # Проверяем, существует ли уже обработанный MACHINE_ файл
    if (Test-Path $machineFilePath) {
        Write-Host "Plotting for: $($file.Name)" -ForegroundColor Cyan
        
        # Запуск Python скрипта для построения графика (передаем оба файла)
        python $SCRIPT_PATH $machineFilePath --annotation-file $file.FullName --patterns $PATTERNS_PATH --output-dir $OUTPUT_DIR
    } else {
        Write-Host "Skipping $($file.Name): Matching MACHINE_ file not found in output directory." -ForegroundColor Yellow
    }
}

Write-Host "All plots generated successfully!" -ForegroundColor Green
