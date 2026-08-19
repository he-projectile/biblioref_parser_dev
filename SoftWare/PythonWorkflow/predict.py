import argparse
from pathlib import Path

import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)


# ============================================================
# CONFIG
# ============================================================

MAX_LENGTH = 512
STRIDE = 128

BIB_LABEL = "БИБЛ. ССЫЛКА"


# ============================================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================================

def load_model(model_dir):

    print()
    print("=" * 70)
    print("ЗАГРУЗКА МОДЕЛИ")
    print("=" * 70)

    print(
        f"Модель: {model_dir}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_dir
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)

    model.eval()

    print(
        f"Device: {device}"
    )

    print(
        f"Labels: {model.config.id2label}"
    )

    return (
        tokenizer,
        model,
        device
    )


# ============================================================
# ЗАГРУЗКА TXT
# ============================================================

def load_text(path):

    path = Path(path)

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        text = f.read()

    print()
    print(
        f"TXT: {path}"
    )

    print(
        f"Длина текста: {len(text)} символов"
    )

    return text


# ============================================================
# TOKENIZE
# ============================================================

def tokenize_text(
    text,
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

    return encoding


# ============================================================
# PREDICTION
# ============================================================

def predict_windows(
    encoding,
    model,
    device
):

    predictions = []

    probabilities = []

    offsets_all = encoding[
        "offset_mapping"
    ]

    input_ids_all = encoding[
        "input_ids"
    ]

    attention_masks_all = encoding[
        "attention_mask"
    ]

    print()
    print(
        f"Количество окон: "
        f"{len(input_ids_all)}"
    )

    for window_idx in range(
        len(input_ids_all)
    ):

        input_ids = torch.tensor(
            [
                input_ids_all[
                    window_idx
                ]
            ],
            dtype=torch.long,
            device=device
        )

        attention_mask = torch.tensor(
            [
                attention_masks_all[
                    window_idx
                ]
            ],
            dtype=torch.long,
            device=device
        )

        with torch.no_grad():

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        logits = outputs.logits[0]

        probs = torch.softmax(
            logits,
            dim=-1
        )

        pred_ids = torch.argmax(
            probs,
            dim=-1
        )

        predictions.append(
            pred_ids.cpu().numpy()
        )

        probabilities.append(
            probs.cpu().numpy()
        )

    return (
        predictions,
        probabilities,
        offsets_all
    )


# ============================================================
# TOKEN → SPAN
# ============================================================

def extract_spans(
    predictions,
    probabilities,
    offsets_all,
    model
):

    """
    Преобразует B-BIB / I-BIB обратно
    в символьные интервалы.

    Возвращает:

    {
        start,
        end,
        confidence
    }
    """

    all_spans = []

    id2label = model.config.id2label

    for window_idx in range(
        len(predictions)
    ):

        pred_ids = predictions[
            window_idx
        ]

        probs = probabilities[
            window_idx
        ]

        offsets = offsets_all[
            window_idx
        ]

        current_start = None
        current_end = None
        current_confidences = []

        for token_idx in range(
            len(pred_ids)
        ):

            token_start, token_end = (
                offsets[token_idx]
            )

            # Специальные токены
            if token_start == token_end:

                continue

            label = id2label[
                int(pred_ids[token_idx])
            ]

            confidence = float(
                probs[token_idx][
                    int(pred_ids[token_idx])
                ]
            )

            # ------------------------------------------------
            # BEGIN BIB
            # ------------------------------------------------

            if label == "B-BIB":

                # Если предыдущая сущность
                # ещё не закрыта
                if current_start is not None:

                    all_spans.append(
                        {
                            "start": current_start,

                            "end": current_end,

                            "confidence": (
                                float(
                                    np.mean(
                                        current_confidences
                                    )
                                )
                            ),
                        }
                    )

                current_start = token_start

                current_end = token_end

                current_confidences = [
                    confidence
                ]

            # ------------------------------------------------
            # CONTINUE BIB
            # ------------------------------------------------

            elif label == "I-BIB":

                if current_start is not None:

                    current_end = token_end

                    current_confidences.append(
                        confidence
                    )

            # ------------------------------------------------
            # OUTSIDE
            # ------------------------------------------------

            else:

                if current_start is not None:

                    all_spans.append(
                        {
                            "start": current_start,

                            "end": current_end,

                            "confidence": (
                                float(
                                    np.mean(
                                        current_confidences
                                    )
                                )
                            ),
                        }
                    )

                    current_start = None

                    current_end = None

                    current_confidences = []

        # ----------------------------------------------------
        # Конец окна
        # ----------------------------------------------------

        if current_start is not None:

            all_spans.append(
                {
                    "start": current_start,

                    "end": current_end,

                    "confidence": (
                        float(
                            np.mean(
                                current_confidences
                            )
                        )
                    ),
                }
            )

    return all_spans


# ============================================================
# ОБЪЕДИНЕНИЕ ПЕРЕСЕКАЮЩИХСЯ SPANS
# ============================================================

def merge_overlapping_spans(
    spans
):

    """
    Окна перекрываются.

    Поэтому одна и та же библиографическая
    ссылка может быть предсказана несколько раз.

    Например:

        100-250
        100-250

    или:

        100-230
        120-250

    Объединяем пересечения.
    """

    if not spans:
        return []

    spans = sorted(
        spans,
        key=lambda x: (
            x["start"],
            x["end"]
        )
    )

    merged = []

    current = spans[0].copy()

    current_confidences = [
        current["confidence"]
    ]

    for span in spans[1:]:

        # Есть пересечение
        if span["start"] <= current["end"]:

            current["end"] = max(
                current["end"],
                span["end"]
            )

            current_confidences.append(
                span["confidence"]
            )

            current["confidence"] = float(
                np.mean(
                    current_confidences
                )
            )

        else:

            merged.append(
                current
            )

            current = span.copy()

            current_confidences = [
                current["confidence"]
            ]

    merged.append(
        current
    )

    return merged


# ============================================================
# УДАЛЕНИЕ СЛИШКОМ МАЛЕНЬКИХ SPANS
# ============================================================

def filter_spans(
    spans,
    min_length=3
):

    result = []

    for span in spans:

        if (
            span["end"]
            - span["start"]
            >= min_length
        ):

            result.append(
                span
            )

    return result


# ============================================================
# ДОБАВЛЕНИЕ TEXT
# ============================================================

def attach_text(
    spans,
    text
):

    result = []

    for span in spans:

        start = span["start"]

        end = span["end"]

        item = {
            "start": start,

            "end": end,

            "text": text[
                start:end
            ],

            "confidence": span[
                "confidence"
            ],
        }

        result.append(
            item
        )

    return result


# ============================================================
# ПЕЧАТЬ РЕЗУЛЬТАТА
# ============================================================

def print_results(
    spans
):

    print()
    print("=" * 70)

    print(
        f"НАЙДЕНО БИБЛИОГРАФИЧЕСКИХ "
        f"ССЫЛОК: {len(spans)}"
    )

    print("=" * 70)

    for idx, span in enumerate(
        spans,
        start=1
    ):

        print()
        print(
            f"[{idx}] "
            f"{span['start']}:"
            f"{span['end']}"
        )

        print(
            f"confidence: "
            f"{span['confidence']:.4f}"
        )

        print(
            repr(span["text"])
        )


# ============================================================
# SAVE JSON
# ============================================================

def save_results(
    output_path,
    text_path,
    spans
):

    import json

    result = {

        "document": str(
            text_path
        ),

        "annotations": [

            {
                "start": span[
                    "start"
                ],

                "end": span[
                    "end"
                ],

                "label": BIB_LABEL,

                "text": span[
                    "text"
                ],

                "confidence": span[
                    "confidence"
                ],
            }

            for span in spans
        ]
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=4
        )

    print()
    print(
        f"Результат сохранён: "
        f"{output_path}"
    )


# ============================================================
# MAIN PREDICT
# ============================================================

def predict(
    text_path,
    model_dir,
    output_path=None
):

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    (
        tokenizer,
        model,
        device
    ) = load_model(
        model_dir
    )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    text = load_text(
        text_path
    )

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    print()
    print(
        "Токенизация..."
    )

    encoding = tokenize_text(
        text,
        tokenizer
    )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    print(
        "Выполнение предсказания..."
    )

    (
        predictions,
        probabilities,
        offsets_all
    ) = predict_windows(
        encoding,
        model,
        device
    )

    # --------------------------------------------------------
    # Token → spans
    # --------------------------------------------------------

    print(
        "Преобразование токенов "
        "в библиографические spans..."
    )

    spans = extract_spans(
        predictions,
        probabilities,
        offsets_all,
        model
    )

    print(
        f"Получено сырых spans: "
        f"{len(spans)}"
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    spans = merge_overlapping_spans(
        spans
    )

    print(
        f"После объединения: "
        f"{len(spans)}"
    )

    # --------------------------------------------------------
    # Filter
    # --------------------------------------------------------

    spans = filter_spans(
        spans
    )

    # --------------------------------------------------------
    # Add text
    # --------------------------------------------------------

    spans = attach_text(
        spans,
        text
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_results(
        spans
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if output_path is not None:

        save_results(
            output_path,
            text_path,
            spans
        )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Поиск библиографических ссылок "
            "с помощью обученной XLM-R модели."
        )
    )

    parser.add_argument(
        "text",
        help="TXT файл для анализа"
    )

    parser.add_argument(
        "--model",
        default="bibliography_model",
        help="Каталог обученной модели"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="JSON файл для сохранения результата"
    )

    args = parser.parse_args()

    predict(
        text_path=args.text,

        model_dir=args.model,

        output_path=args.output
    )


if __name__ == "__main__":
    main()