import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch

import os

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# CONFIG
# ============================================================

TOKEN_PATH = r"C:\Users\barko\Desktop\Daniil\MIPT\SRW\SoftWare\PythonWorkflow\secrets\hf_tkn_RO.tkn"

if os.path.exists(TOKEN_PATH):
    with open(TOKEN_PATH, "r", encoding="utf-8") as f:
        hf_token = f.read().strip()

    os.environ["HF_TOKEN"] = hf_token
    print("Токен Hugging Face успешно загружен из файла!")
else:
    print(f"Ошибка: Файл не найден по пути {TOKEN_PATH}")


MODEL_NAME = "xlm-roberta-base"

LABELS = [
    "O",
    "B-BIB",
    "I-BIB",
]

LABEL2ID = {
    label: i
    for i, label in enumerate(LABELS)
}

ID2LABEL = {
    i: label
    for i, label in enumerate(LABELS)
}

BIB_LABEL = "БИБЛ. ССЫЛКА"

MAX_LENGTH = 512
STRIDE = 128


# ============================================================
# SEED
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# ЗАГРУЗКА ОДНОЙ РАЗМЕТКИ
# ============================================================

def load_annotation_file(json_path):

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:

        annotation = json.load(f)

    if "annotations" not in annotation:
        raise ValueError(
            f"{json_path}: "
            f"нет поля 'annotations'"
        )

    if "document" not in annotation:
        raise ValueError(
            f"{json_path}: "
            f"нет поля 'document'"
        )

    return annotation


# ============================================================
# ПОИСК TXT
# ============================================================

def load_document_text(
    annotation,
    json_path
):

    document_name = annotation["document"]

    # Сначала ищем TXT рядом с JSON
    txt_path = (
        json_path.parent
        / document_name
    )

    # Если не нашли — просто ищем по имени
    if not txt_path.exists():

        raise FileNotFoundError(
            f"Не найден TXT для:\n"
            f"{json_path}\n\n"
            f"Ожидался:\n"
            f"{txt_path}"
        )

    with open(
        txt_path,
        "r",
        encoding="utf-8"
    ) as f:

        text = f.read()

    return text, txt_path


# ============================================================
# ПОЛУЧЕНИЕ БИБЛИОГРАФИЧЕСКИХ ССЫЛОК
# ============================================================

def get_bibliography_spans(
    annotations
):

    spans = []

    for annotation in annotations:

        if annotation.get("label") != BIB_LABEL:
            continue

        start = int(
            annotation["start"]
        )

        end = int(
            annotation["end"]
        )

        if end <= start:
            continue

        spans.append(
            (
                start,
                end
            )
        )

    spans.sort()

    return spans


# ============================================================
# ПРОВЕРКА КООРДИНАТ
# ============================================================

def validate_spans(
    text,
    spans,
    source
):

    text_length = len(text)

    for start, end in spans:

        if start < 0:

            raise ValueError(
                f"{source}: "
                f"start={start} < 0"
            )

        if end > text_length:

            raise ValueError(
                f"{source}: "
                f"end={end} > "
                f"text length={text_length}"
            )

        if start >= end:

            raise ValueError(
                f"{source}: "
                f"bad span "
                f"{start}:{end}"
            )

        # Очень полезная проверка:
        # показываем первые символы реального span.
        print(
            f"    span "
            f"{start}:{end}: "
            f"{repr(text[start:end][:80])}"
        )


# ============================================================
# ТОКЕНИЗАЦИЯ + ALIGNMENT
# ============================================================

def tokenize_and_align(
    text,
    spans,
    tokenizer
):

    encoding = tokenizer(
        text,

        truncation=True,

        max_length=MAX_LENGTH,

        stride=STRIDE,

        return_overflowing_tokens=True,

        return_offsets_mapping=True,

        padding=False,
    )

    examples = []

    for window_idx in range(
        len(encoding["input_ids"])
    ):

        input_ids = encoding[
            "input_ids"
        ][window_idx]

        attention_mask = encoding[
            "attention_mask"
        ][window_idx]

        offsets = encoding[
            "offset_mapping"
        ][window_idx]

        labels = []

        for token_start, token_end in offsets:

            # Специальный токен:
            # <s>, </s> и т.п.
            if token_start == token_end:

                labels.append(-100)

                continue

            label = "O"

            for span_start, span_end in spans:

                # Нет пересечения
                if (
                    token_end <= span_start
                    or token_start >= span_end
                ):
                    continue

                # Токен пересекает начало entity
                if (
                    token_start
                    <= span_start
                    < token_end
                ):

                    label = "B-BIB"

                else:

                    label = "I-BIB"

                break

            labels.append(
                LABEL2ID[label]
            )

        examples.append(
            {
                "input_ids": input_ids,

                "attention_mask": attention_mask,

                "labels": labels,

                "offset_mapping": offsets,
            }
        )

    return examples


# ============================================================
# ПОДГОТОВКА ДОКУМЕНТА
# ============================================================

def prepare_document(
    annotation,
    text,
    tokenizer,
    document_id
):

    annotations = annotation[
        "annotations"
    ]

    spans = get_bibliography_spans(
        annotations
    )

    print()
    print(
        f"Документ {document_id}:"
    )

    print(
        f"  Символов: {len(text)}"
    )

    print(
        f"  Библиографических ссылок: "
        f"{len(spans)}"
    )

    validate_spans(
        text,
        spans,
        annotation["document"]
    )

    if not spans:

        return []

    examples = tokenize_and_align(
        text,
        spans,
        tokenizer
    )

    for example in examples:

        example["document_id"] = document_id

    return examples


# ============================================================
# ЗАГРУЗКА ВСЕГО DATASET
# ============================================================

def load_dataset_from_directory(
    data_dir,
    tokenizer
):

    data_dir = Path(data_dir)

    json_files = sorted(
        data_dir.glob("*.json")
    )

    if not json_files:

        raise ValueError(
            f"В {data_dir} нет JSON файлов."
        )

    print(
        f"Найдено JSON файлов: "
        f"{len(json_files)}"
    )

    examples = []

    documents = []

    for document_id, json_path in enumerate(
        json_files
    ):

        print()
        print("=" * 70)

        print(
            f"JSON: {json_path.name}"
        )

        annotation = load_annotation_file(
            json_path
        )

        text, txt_path = load_document_text(
            annotation,
            json_path
        )

        print(
            f"TXT: {txt_path.name}"
        )

        documents.append(
            {
                "annotation": annotation,
                "text": text,
                "json_path": str(
                    json_path
                ),
                "txt_path": str(
                    txt_path
                ),
            }
        )

    return documents


def has_bibliography(annotation):
    """
    Есть ли в документе хотя бы одна
    размеченная БИБЛ. ССЫЛКА?
    """

    for entity in annotation.get(
        "annotations",
        []
    ):

        if entity.get("label") == BIB_LABEL:
            return True

    return False

def count_bibliography(
    documents
):
    return sum(
        len(
            get_bibliography_spans(
                document["annotation"]["annotations"]
            )
        )
        for document in documents
    )

# ============================================================
# SPLIT ПО ДОКУМЕНТАМ
# ============================================================

def split_documents(
    documents,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42
):
    """
    Делит документы, а не окна.

    При небольшом количестве документов
    используем минимум по одному документу
    для validation и test.
    """

    documents = list(documents)

    if len(documents) < 3:
        raise ValueError(
            "Для train/validation/test нужно "
            "минимум 3 документа."
        )

    rng = random.Random(seed)
    rng.shuffle(documents)

    n = len(documents)

    # Минимум по одному документу
    # в validation и test.
    n_test = max(
        1,
        round(n * 0.1)
    )

    n_val = max(
        1,
        round(n * 0.1)
    )

    # Не даём validation + test съесть весь dataset
    if n_val + n_test >= n:
        n_val = 1
        n_test = 1

    n_train = n - n_val - n_test

    train_documents = documents[
        :n_train
    ]

    val_documents = documents[
        n_train:n_train + n_val
    ]

    test_documents = documents[
        n_train + n_val:
    ]

    return (
        train_documents,
        val_documents,
        test_documents
    )


# ============================================================
# PREPARE SPLIT
# ============================================================

def prepare_split(
    documents,
    tokenizer,
    split_name
):

    print()
    print("=" * 70)

    print(
        f"ПОДГОТОВКА {split_name.upper()}"
    )

    print("=" * 70)

    examples = []

    for document_id, document in enumerate(
        documents
    ):

        new_examples = prepare_document(
            annotation=document["annotation"],

            text=document["text"],

            tokenizer=tokenizer,

            document_id=document_id,
        )

        examples.extend(
            new_examples
        )

    print()
    print(
        f"{split_name}: "
        f"{len(examples)} windows"
    )

    if not examples:

        raise ValueError(
            f"Split {split_name} "
            f"не содержит обучающих примеров."
        )

    return Dataset.from_list(
        examples
    )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    eval_prediction
):

    predictions, labels = eval_prediction

    predictions = np.argmax(
        predictions,
        axis=2
    )

    true_predictions = []

    true_labels = []

    for prediction, label in zip(
        predictions,
        labels
    ):

        current_predictions = []

        current_labels = []

        for pred, lab in zip(
            prediction,
            label
        ):

            if lab == -100:
                continue

            current_predictions.append(
                ID2LABEL[int(pred)]
            )

            current_labels.append(
                ID2LABEL[int(lab)]
            )

        true_predictions.append(
            current_predictions
        )

        true_labels.append(
            current_labels
        )

    precision = precision_score(
        true_labels,
        true_predictions
    )

    recall = recall_score(
        true_labels,
        true_predictions
    )

    f1 = f1_score(
        true_labels,
        true_predictions
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# TRAIN
# ============================================================

def train(
    data_dir,
    output_dir,
    epochs,
    batch_size,
    learning_rate,
    seed
):

    set_seed(seed)

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    print()
    print(
        f"Загрузка tokenizer: "
        f"{MODEL_NAME}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    documents = load_dataset_from_directory(
        data_dir,
        tokenizer
    )

    documents = [
        document
        for document in documents
        if has_bibliography(
            document["annotation"]
        )
    ]

    print()
    print(
        f"Всего документов: "
        f"{len(documents)}"
    )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    (
        train_documents,
        val_documents,
        test_documents
    ) = split_documents(
        documents,
        seed=seed
    )

    print()
    print("=" * 70)
    print("РАЗБИЕНИЕ DATASET")
    print("=" * 70)

    print(
        f"Train: "
        f"{len(train_documents)} документов, "
        f"{count_bibliography(train_documents)} ссылок"
    )

    print(
        f"Validation: "
        f"{len(val_documents)} документов, "
        f"{count_bibliography(val_documents)} ссылок"
    )

    print(
        f"Test: "
        f"{len(test_documents)} документов, "
        f"{count_bibliography(test_documents)} ссылок"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = prepare_split(
        train_documents,
        tokenizer,
        "train"
    )

    val_dataset = prepare_split(
        val_documents,
        tokenizer,
        "validation"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print(
        f"Загрузка модели: "
        f"{MODEL_NAME}"
    )

    model = AutoModelForTokenClassification.from_pretrained(

        MODEL_NAME,

        num_labels=len(LABELS),

        id2label=ID2LABEL,

        label2id=LABEL2ID,
    )

    # --------------------------------------------------------
    # COLLATOR
    # --------------------------------------------------------

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    # --------------------------------------------------------
    # TRAINING ARGS
    # --------------------------------------------------------

    training_args = TrainingArguments(

        output_dir=output_dir,

        num_train_epochs=epochs,

        per_device_train_batch_size=batch_size,

        per_device_eval_batch_size=batch_size,

        learning_rate=learning_rate,

        weight_decay=0.01,

        logging_steps=10,

        eval_strategy="epoch",

        save_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model="f1",

        greater_is_better=True,

        save_total_limit=2,

        report_to="none",

        fp16=torch.cuda.is_available(),

        seed=seed,
    )

    # --------------------------------------------------------
    # TRAINER
    # --------------------------------------------------------

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=val_dataset,

        processing_class=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics,
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "НАЧАЛО ОБУЧЕНИЯ"
    )

    print("=" * 70)

    trainer.train()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "VALIDATION"
    )

    print("=" * 70)

    metrics = trainer.evaluate()

    for key, value in metrics.items():

        if isinstance(
            value,
            float
        ):

            print(
                f"{key}: "
                f"{value:.4f}"
            )

        else:

            print(
                f"{key}: "
                f"{value}"
            )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    print()
    print(
        f"Сохранение модели: "
        f"{output_dir}"
    )

    trainer.save_model(
        output_dir
    )

    tokenizer.save_pretrained(
        output_dir
    )

    # --------------------------------------------------------
    # SAVE TEST SET
    # --------------------------------------------------------

    test_file = (
        Path(output_dir)
        / "test_documents.json"
    )

    with open(
        test_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            [
                {
                    "json_path": d[
                        "json_path"
                    ],

                    "txt_path": d[
                        "txt_path"
                    ],

                    "document": d[
                        "annotation"
                    ]["document"],
                }

                for d in test_documents
            ],

            f,

            ensure_ascii=False,

            indent=2
        )

    print()
    print(
        "Обучение завершено."
    )

    print(
        f"Модель: {output_dir}"
    )

    print(
        f"Test: {test_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data",
        help=(
            "Каталог с JSON + TXT "
            "разметкой"
        )
    )

    parser.add_argument(
        "--output",
        default="bibliography_model"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    train(
        data_dir=args.data,

        output_dir=args.output,

        epochs=args.epochs,

        batch_size=args.batch_size,

        learning_rate=args.learning_rate,

        seed=args.seed,
    )


if __name__ == "__main__":
    main()