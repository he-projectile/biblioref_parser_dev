import os
import json
import re

# Укажите путь к вашей папке с данными (из args.data)
DATA_DIR = r"C:\Users\barko\Desktop\Daniil\MIPT\SRW\SoftWare\PythonWorkflow\DataSource" 

def clean_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Сначала пробуем прочитать мягко
    try:
        data = json.loads(content, strict=False)
    except json.JSONDecodeError:
        # 2. Если не вышло, удаляем жесткие битые символы
        content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", content)
        data = json.loads(content, strict=False)
        
    # Перезаписываем файл уже в идеальном, чистом JSON-формате
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Обходим все файлы в папке
count = 0
for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith('.json'):
            try:
                clean_json_file(os.path.join(root, file))
                count += 1
            except Exception as e:
                print(f"Не удалось починить файл {file}: {e}")

print(f"Готово! Успешно очищено файлов: {count}")
