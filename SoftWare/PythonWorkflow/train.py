import json
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


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


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# Загрузка твоей разметки
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# Преобразование span-разметки
# ============================================================

def prepare_example(example):

    text = example["text"]

    entities = [
        e
        for e in example["entities"]
        if e["label"] == "БИБЛ. ССЫЛКА"
    ]

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
    )

    offsets = encoding["offset_mapping"]

    labels = []

    for token_start, token_end in offsets:

        # Специальный токен
        if token_start == token_end:

            labels.append(-100)
            continue

        label = "O"

        for entity in entities:

            start = entity["start"]
            end = entity["end"]

            # Есть пересечение токена и entity
            if (
                token_start < end
                and token_end > start
            ):

                if token_start <= start:
                    label = "B-BIB"
                else:
                    label = "I-BIB"

                break

        labels.append(
            LABEL2ID[label]
        )

    encoding["labels"] = labels

    # offset_mapping больше не нужен Trainer
    del encoding["offset_mapping"]

    return encoding