import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix

def load_pair_data(data_dir):
    """Загружает пары .txt + .json из указанной директории и размечает строки."""
    documents = []
    if not os.path.exists(data_dir):
        print(f"❌ Ошибка: Папка {data_dir} не существует!")
        return []
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    for txt_file in files:
        base_name = os.path.splitext(txt_file)[0]
        json_file = f"{base_name}.json"
        txt_path = os.path.join(data_dir, txt_file)
        json_path = os.path.join(data_dir, json_file)
        
        if not os.path.exists(json_path): 
            continue
        with open(txt_path, 'r', encoding='utf-8') as f: text = f.read()
        with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
            
        char_labels = np.zeros(len(text), dtype=int)
        annotations = data.get('annotations', [])
        for ann in annotations:
            if 'children' in ann and ann['children']:
                starts = [child['start'] for child in ann['children'] if 'start' in child]
                ends = [child['end'] for child in ann['children'] if 'end' in child]
                if starts and ends: char_labels[max(0, min(starts)):min(len(text), max(ends))] = 1
            elif 'start' in ann and 'end' in ann:
                char_labels[max(0, ann['start']):min(len(text), ann['end'])] = 1

        lines_data = []
        lines = text.split('\n')
        accumulated_idx = 0
        for line in lines:
            line_len = len(line)
            line_char_labels = char_labels[accumulated_idx : accumulated_idx + line_len]
            if len(line_char_labels) > 0 and (line_char_labels.sum() / len(line_char_labels)) > 0.1:
                is_ref = 1
            else:
                is_ref = 0
            lines_data.append((line, is_ref))
            accumulated_idx += line_len + 1
            
        if lines_data: 
            documents.append({'filename': txt_file, 'lines': lines_data})
            
    print(f"📁 Из папки [{os.path.basename(data_dir)}] успешно загружено документов: {len(documents)}")
    return documents

def compute_macro_block_iou(y_true, y_pred):
    """Расчет IoU непрерывных интервалов макро-блоков."""
    true_indices = np.where(y_true == 1)[0]
    pred_indices = np.where(y_pred == 1)[0]
    if len(true_indices) == 0 and len(pred_indices) == 0: return 1.0
    if len(true_indices) == 0 or len(pred_indices) == 0: return 0.0
    
    t_start, t_end = true_indices[0], true_indices[-1]
    p_start, p_end = pred_indices[0], pred_indices[-1]
    
    intersection_start = max(t_start, p_start)
    intersection_end = min(t_end, p_end)
    intersection = max(0, intersection_end - intersection_start + 1)
    union = max(t_end, p_end) - min(t_start, p_start) + 1
    return intersection / union

def find_best_sequence(probabilities, threshold=0.45):
    """Поиск границ плотного макро-кластера (алгоритм Кадана)."""
    binary_preds = (probabilities > threshold).astype(int)
    if binary_preds.sum() == 0: return binary_preds

    n = len(binary_preds)
    best_start, best_end = 0, 0
    max_score = -1
    current_score = 0
    start_track = 0
    
    for i in range(n):
        step_score = 2 if binary_preds[i] == 1 else -3
        if current_score == 0 and step_score > 0:
            start_track = i
        current_score = max(0, current_score + step_score)
        
        if current_score > max_score:
            max_score = current_score
            best_start = start_track
            best_end = i
            
    smoothed = np.zeros_like(binary_preds)
    if max_score > 5:
        sub_window = binary_preds[best_start:best_end+1]
        ones = np.where(sub_window == 1)[0]
        if len(ones) > 0:
            smoothed[best_start + ones[0] : best_start + ones[-1] + 1] = 1
            
    return smoothed

def plot_iou_histogram(train_ious, test_ious, output_filename):
    """Отрисовка распределения IoU строго по переданному шаблону стиля (Train и Test)."""
    bins = np.linspace(0.0, 1.0, 21)

    datasets = [
        ("Train (opt.)",    train_ious, "#1f77b4"),
        ("Test (held-out)", test_ious,  "#2ca02c"),
    ]

    active_datasets = [d for d in datasets if len(d[1]) > 0]
    num_plots = len(active_datasets)

    if num_plots == 0:
        return

    fig, axes = plt.subplots(
        num_plots, 1, figsize=(10, 3 * num_plots), sharex=True, sharey=True
    )

    if num_plots == 1:
        axes = [axes]

    for ax, (name, ious, color) in zip(axes, active_datasets):
        weights = np.ones(len(ious), dtype=float) / len(ious)

        mean_val = np.mean(ious)
        median_val = np.median(ious)        

        n, bins_out, patches = ax.hist(
            ious,
            bins=bins,
            weights=weights,
            color=color,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.5,
        )

        lineMean = ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        lineMedian  = ax.axvline(median_val, color='darkgreen', linestyle=':', linewidth=2, alpha=0.8)

        ax.set_ylabel("Доля документов")
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, linestyle="--", alpha=0.5)

        legend_text = [
            f"{name} (n={len(ious)})\n",
            f"Mean: {mean_val:.3f}\n",
            f"Median: {median_val:.3f}"
        ]

        ax.legend(
            handles=[patches, lineMean, lineMedian],
            labels=legend_text,
            loc="upper left", 
            frameon=True
        )

    axes[-1].set_xlabel("IoU")

    plt.suptitle(
        "Распределение IoU по поддатасетам (ML-Бейзлайн)\n"
        "(веса выбраны по Train, Test — отложенная оценка)",
        fontsize=12, y=0.99
    )
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches="tight")
    plt.close()

def generate_split_html_report(train_results, test_results, output_path="baseline_split_report.html"):
    mean_train_iou = np.mean([d['iou'] for d in train_results]) if train_results else 0
    mean_test_iou = np.mean([d['iou'] for d in test_results]) if test_results else 0
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Раздельная Валидация БР (Train/Test)</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 30px; background-color: #f4f6f9; }}
            .dashboard {{ display: flex; gap: 20px; margin-bottom: 30px; }}
            .metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); flex: 1; }}
            .metric-value {{ font-size: 28px; font-weight: bold; }}
            .train-color {{ color: #2b579a; }}
            .test-color {{ color: #cc0000; }}
            .section-title {{ border-bottom: 2px solid #ccc; padding-bottom: 5px; margin-top: 25px; }}
            .doc-section {{ background: white; padding: 15px; border-radius: 8px; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
            .line-container {{ font-family: monospace; white-space: pre-wrap; font-size: 13px; line-height: 1.4; padding: 1px 5px; }}
            .macro-overlap {{ border-left: 6px solid #ffc107; border-right: 6px solid #28a745; background-color: #d4edda; }}
            .macro-true-block {{ border-left: 6px solid #ffc107; background-color: #fff9e6; }}
            .macro-pred-block {{ border-right: 6px solid #28a745; background-color: #e6f4ea; }}
            .tn {{ color: #aaa; }}
        </style>
    </head>
    <body>
        <h1>📊 Сравнительный отчет локализации БР: Обучение vs Тест</h1>
        <div class="dashboard">
            <div class="metric-card">
                <div>Средний Macro IoU на <b>Обучении (Train)</b>:</div>
                <div class="metric-value train-color">{mean_train_iou:.4f}</div>
            </div>
            <div class="metric-card">
                <div>Средний Macro IoU на <b>Тесте (Test)</b>:</div>
                <div class="metric-value test-color">{mean_test_iou:.4f}</div>
            </div>
        </div>
    """
    
    def render_section(results, title_text):
        section_html = f"<h2 class='section-title'>{title_text}</h2>"
        for doc in results:
            section_html += f"""
            <div class="doc-section">
                <h4>📄 {doc['filename']} | Macro IoU: <span style="color: {'#28a745' if doc['iou'] > 0.7 else '#dc3545'}">{doc['iou']:.4f}</span></h4>
                <div style="max-height: 250px; overflow-y: auto; border: 1px solid #ddd; padding: 8px; border-radius: 5px;">
            """
            for i, (line, y_t, y_p) in enumerate(zip(doc['lines_text'], doc['y_true'], doc['y_pred'])):
                context = 7
                start_c = max(0, i - context)
                end_c = min(len(doc['y_true']), i + context + 1)
                if not any(doc['y_true'][start_c:end_c]) and not any(doc['y_pred'][start_c:end_c]): continue
                
                cls = "tn"
                meta = ""
                if y_t == 1 and y_p == 1: cls, meta = "macro-overlap", "[БР]"
                elif y_t == 1 and y_p == 0: cls, meta = "macro-true-block", "[ПРОПУСК]"
                elif y_t == 0 and y_p == 1: cls, meta = "macro-pred-block", "[ЛОЖНО]"
                
                if line.strip() == "": line = "&nbsp;"
                section_html += f'<div class="line-container {cls}">{meta} {line}</div>'
            section_html += "</div></div>"
        return section_html

    html_content += render_section(train_results, "📘 ОБУЧАЮЩАЯ ВЫБОРКА (TRAIN DATA)")
    html_content += render_section(test_results, "📕 ТЕСТОВАЯ ВЫБОРКА (TEST DATA)")
    html_content += "</body></html>"
    
    with open(output_path, 'w', encoding='utf-8') as f: 
        f.write(html_content)
    print(f"✨ Раздельный HTML-отчет успешно сохранен в: {output_path}")

def run_split_pipeline(train_dir, test_dir):
    train_docs = load_pair_data(train_dir)
    test_docs = load_pair_data(test_dir)
    
    if not train_docs:
        print("❌ Нет обучающих данных! Пайплайн остановлен.")
        return
        
    # 1. Сбор признаков для Train
    X_train_text, X_train_spatial, y_train = [], [], []
    for doc in train_docs:
        total_lines = len(doc['lines'])
        for idx, (line, label) in enumerate(doc['lines']):
            X_train_text.append(line)
            rel_pos = idx / total_lines
            X_train_spatial.append([rel_pos, 1.0 if rel_pos > 0.75 else 0.0])
            y_train.append(label)
            
    # 2. Обучение базового стабильного TF-IDF + Логистической регрессии
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=10000)
    X_train_full = hstack([vectorizer.fit_transform(X_train_text), csr_matrix(X_train_spatial)])
    
    clf = LogisticRegression(C=2.0, class_weight='balanced', max_iter=1000, random_state=42)
    clf.fit(X_train_full, y_train)
    print("🎯 Базовая модель бейзлайна успешно обучена.")
    
    # 3. Инференс на Train
    train_results = []
    for doc in train_docs:
        X_doc_text, X_doc_spatial = [], []
        y_doc_true = np.array([label for _, label in doc['lines']])
        total_lines = len(doc['lines'])
        for idx, (line, _) in enumerate(doc['lines']):
            X_doc_text.append(line)
            rel_pos = idx / total_lines
            X_doc_spatial.append([rel_pos, 1.0 if rel_pos > 0.75 else 0.0])
            
        X_doc_full = hstack([vectorizer.transform(X_doc_text), csr_matrix(X_doc_spatial)])
        probs = clf.predict_proba(X_doc_full)[:, 1]
        y_doc_pred = find_best_sequence(probs)
        
        train_results.append({
            'filename': doc['filename'], 
            'lines_text': X_doc_text,
            'y_true': y_doc_true.tolist(), 
            'y_pred': y_doc_pred.tolist(),
            'iou': compute_macro_block_iou(y_doc_true, y_doc_pred)
        })
        
    # 4. Инференс на Test
    test_results = []
    for doc in test_docs:
        X_doc_text, X_doc_spatial = [], []
        y_doc_true = np.array([label for _, label in doc['lines']])
        total_lines = len(doc['lines'])
        for idx, (line, _) in enumerate(doc['lines']):
            X_doc_text.append(line)
            rel_pos = idx / total_lines
            X_doc_spatial.append([rel_pos, 1.0 if rel_pos > 0.75 else 0.0])
            
        X_doc_full = hstack([vectorizer.transform(X_doc_text), csr_matrix(X_doc_spatial)])
        probs = clf.predict_proba(X_doc_full)[:, 1]
        y_doc_pred = find_best_sequence(probs)
        
        test_results.append({
            'filename': doc['filename'], 
            'lines_text': X_doc_text,
            'y_true': y_doc_true.tolist(), 
            'y_pred': y_doc_pred.tolist(),
            'iou': compute_macro_block_iou(y_doc_true, y_doc_pred)
        })
        
    train_ious = [d['iou'] for d in train_results]
    test_ious = [d['iou'] for d in test_results]
    
    print(f"\n📈 Результаты -> Train Mean IoU: {np.mean(train_ious):.4f} | Test Mean IoU: {np.mean(test_ious):.4f}")
    
    # Генерация графиков распределения
    plot_iou_histogram(train_ious, test_ious, "baseline_iou_distribution.png")
    print("🎨 Двухслойная гистограмма распределения сохранена в: baseline_iou_distribution.png")
    
    # Генерация HTML-отчета
    generate_split_html_report(train_results, test_results)

if __name__ == '__main__':
    TRAIN_DIR = r"C:\Users\Barkock\Desktop\Личное\biblioRef\SoftWare\PythonWorkflow\DataSource"
    TEST_DIR = r"C:\Users\Barkock\Desktop\Личное\biblioRef\SoftWare\PythonWorkflow\DataSource\Test"
    
    run_split_pipeline(TRAIN_DIR, TEST_DIR)
