import argparse
import json
import math
import re
import random
from collections import Counter, defaultdict
from pathlib import Path


REFERENCE_LABEL = "БИБЛ. ССЫЛКА"


# ============================================================
# REGEX-КЛАССЫ
# ============================================================

URL_RE = re.compile(
    r"https?://[^\s]+",
    re.IGNORECASE
)

DOI_RE = re.compile(
    r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b",
    re.IGNORECASE
)

ISBN_RE = re.compile(
    r"\bISBN(?:[- ]?(?:10|13))?[- ]?"
    r"(?:97[89][- ]?)?"
    r"[\dXx][\dXx -]{8,20}\b",
    re.IGNORECASE
)

ISSN_RE = re.compile(
    r"\b\d{4}[-–—]\d{3}[\dXx]\b",
    re.IGNORECASE
)

YEAR_RE = re.compile(
    r"(?<!\d)(?:18|19|20)\d{2}(?!\d)"
)

NUMBER_RE = re.compile(
    r"\d+(?:[.,]\d+)?"
)

WORD_RE = re.compile(
    r"[A-Za-zА-Яа-яЁё]+"
)

INITIAL_RE = re.compile(
    r"[А-ЯЁA-Z]\."
)


# ============================================================
# ЗАГРУЗКА
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_document(json_path):
    """
    Загружает txt + json.
    JSON должен иметь формат:

    {
        "annotations": [
            {
                "start": ...,
                "end": ...,
                "label": "БИБЛ. ССЫЛКА",
                "text": ...
            }
        ]
    }
    """

    txt_path = json_path.with_suffix(".txt")

    if not txt_path.exists():
        return None

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    data = load_json(json_path)

    references = []

    for annotation in data.get("annotations", []):

        if annotation.get("label") != REFERENCE_LABEL:
            continue

        if "start" not in annotation or "end" not in annotation:
            continue

        start = int(annotation["start"])
        end = int(annotation["end"])

        if start < 0 or end <= start or end > len(text):
            continue

        reference_text = text[start:end]

        references.append({
            "start": start,
            "end": end,
            "text": reference_text,
            "children": annotation.get("children", [])
        })

    return {
        "name": json_path.name,
        "text": text,
        "references": references
    }


# ============================================================
# НОРМАЛИЗАЦИЯ
# ============================================================

def normalize_text(text):
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\t", " ")

    # OCR может давать большое количество пробелов
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# ФОРМАЛЬНАЯ ТОКЕНИЗАЦИЯ
# ============================================================

def formal_tokens(text):
    """
    Превращает ссылку в последовательность формальных классов.

    Например:

    Иванов И.И. Название книги. М.: Наука, 2020. 123 с.

    ->
    
    WORD INITIAL INITIAL PUNCT WORD WORD PUNCT
    WORD PUNCT WORD PUNCT WORD PUNCT YEAR PUNCT
    NUMBER WORD PUNCT
    """

    text = normalize_text(text)

    tokens = []

    pos = 0

    token_re = re.compile(
        r"https?://[^\s]+"
        r"|10\.\d{4,9}/[-._;()/:A-Z0-9]+"
        r"|\bISBN(?:[- ]?(?:10|13))?[- ]?(?:97[89][- ]?)?[\dXx][\dXx -]{8,20}"
        r"|\b\d{4}[-–—]\d{3}[\dXx]\b"
        r"|(?:18|19|20)\d{2}"
        r"|\d+(?:[.,]\d+)?"
        r"|[A-Za-zА-Яа-яЁё]+"
        r"|[^\w\s]",
        re.IGNORECASE
    )

    for match in token_re.finditer(text):

        token = match.group()

        # Пропущенный текст между токенами
        pos = match.end()

        # ----------------------------------------------------
        # URL
        # ----------------------------------------------------

        if URL_RE.fullmatch(token):
            tokens.append("URL")
            continue

        # ----------------------------------------------------
        # DOI
        # ----------------------------------------------------

        if DOI_RE.fullmatch(token):
            tokens.append("DOI")
            continue

        # ----------------------------------------------------
        # ISBN
        # ----------------------------------------------------

        if ISBN_RE.fullmatch(token):
            tokens.append("ISBN")
            continue

        # ----------------------------------------------------
        # ISSN
        # ----------------------------------------------------

        if ISSN_RE.fullmatch(token):
            tokens.append("ISSN")
            continue

        # ----------------------------------------------------
        # ГОД
        # ----------------------------------------------------

        if YEAR_RE.fullmatch(token):
            tokens.append("YEAR")
            continue

        # ----------------------------------------------------
        # ЧИСЛО
        # ----------------------------------------------------

        if NUMBER_RE.fullmatch(token):
            tokens.append("NUMBER")
            continue

        # ----------------------------------------------------
        # СЛОВО
        # ----------------------------------------------------

        if WORD_RE.fullmatch(token):

            # Различаем условно "фамилию с инициалами"
            # и обычное слово.
            #
            # Это не семантическая разметка.
            # Это исключительно формальный признак.

            if (
                len(token) <= 2
                and token[0].isupper()
            ):
                tokens.append("SHORT_WORD")
            else:
                tokens.append("WORD")

            continue

        # ----------------------------------------------------
        # ПУНКТУАЦИЯ
        # ----------------------------------------------------

        tokens.append(f"PUNCT:{token}")

    return tuple(tokens)


# ============================================================
# БОЛЕЕ ГРУБАЯ СИГНАТУРА
# ============================================================

def shape_tokens(text):
    """
    Ещё более обобщённая сигнатура.

    WORD      -> W
    NUMBER    -> N
    YEAR      -> Y
    URL       -> U
    DOI       -> D
    punctuation сохраняется.

    Это позволяет проверить:

    насколько далеко можно обобщить
    библиографическую структуру,
    не используя смысл текста.
    """

    tokens = formal_tokens(text)

    result = []

    for token in tokens:

        if token == "YEAR":
            result.append("Y")

        elif token == "NUMBER":
            result.append("N")

        elif token == "URL":
            result.append("U")

        elif token == "DOI":
            result.append("D")

        elif token == "ISBN":
            result.append("ISBN")

        elif token == "ISSN":
            result.append("ISSN")

        elif token == "WORD":
            result.append("W")

        elif token == "SHORT_WORD":
            result.append("S")

        elif token.startswith("PUNCT:"):
            result.append(token)

        else:
            result.append(token)

    return tuple(result)


# ============================================================
# PREFIX-ПАТТЕРН
# ============================================================

def prefix_signature(signature, length):
    """
    Первые N формальных токенов.

    Полезно для будущего поиска начала библиографической
    ссылки.
    """

    if len(signature) <= length:
        return signature

    return signature[:length]


# ============================================================
# СТРУКТУРНЫЙ ПАТТЕРН ИЗ GOLD-РАЗМЕТКИ
# ============================================================

def semantic_signature(reference):
    """
    Используется ТОЛЬКО как исследовательский upper bound.

    Это не формальный locator, потому что для его применения
    в реальном тексте сначала нужно уметь определить поля.
    """

    children = reference.get("children", [])

    children = [
        child
        for child in children
        if "start" in child
        and "end" in child
        and "label" in child
    ]

    children.sort(
        key=lambda x: (
            int(x["start"]),
            int(x["end"])
        )
    )

    return tuple(
        child["label"]
        for child in children
    )


# ============================================================
# ДОКУМЕНТЫ
# ============================================================

def load_dataset(data_dir):

    documents = []

    for json_path in sorted(data_dir.glob("*.json")):

        document = load_document(json_path)

        if document is None:
            print(
                f"[WARNING] Нет txt для {json_path.name}"
            )
            continue

        if not document["references"]:
            continue

        documents.append(document)

    return documents


# ============================================================
# TRAIN / VALIDATION / TEST
# ============================================================

def make_split(documents, seed=42):
    """
    Деление именно по документам.

    Это критично.

    Нельзя допускать ситуацию, когда ссылки из одного документа
    одновременно находятся в train и test.
    """

    names = [
        document["name"]
        for document in documents
    ]

    names = sorted(names)

    rng = random.Random(seed)
    rng.shuffle(names)

    n = len(names)

    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    return {
        "train": names[:train_end],
        "validation": names[train_end:valid_end],
        "test": names[valid_end:]
    }


def split_documents(documents, split):

    result = {
        "train": [],
        "validation": [],
        "test": []
    }

    by_name = {
        document["name"]: document
        for document in documents
    }

    for subset, names in split.items():

        for name in names:

            if name in by_name:
                result[subset].append(
                    by_name[name]
                )

    return result


# ============================================================
# ЭКЗЕМПЛЯРЫ
# ============================================================

def make_instances(documents):

    instances = []

    for document in documents:

        for index, reference in enumerate(
            document["references"]
        ):

            text = reference["text"]

            instances.append({
                "document": document["name"],
                "reference_index": index,
                "text": text,

                "formal": formal_tokens(text),
                "shape": shape_tokens(text),

                "semantic":
                    semantic_signature(reference)
            })

    return instances


# ============================================================
# ПОДСЧЁТ ПАТТЕРНОВ
# ============================================================

def count_patterns(instances, key):

    counter = Counter()

    for item in instances:

        pattern = item[key]

        counter[pattern] += 1

    return counter


# ============================================================
# ПОКРЫТИЕ
# ============================================================

def cumulative_coverage(
    patterns,
    instances,
    key
):
    """
    Для top-K паттернов вычисляет:

        coverage =
        количество уникальных ссылок,
        покрываемых хотя бы одним паттерном
        /
        общее количество ссылок
    """

    if not instances:
        return []

    total = len(instances)

    matched = set()

    result = []

    for k, pattern in enumerate(
        patterns,
        start=1
    ):

        for index, item in enumerate(instances):

            if index in matched:
                continue

            if item[key] == pattern:
                matched.add(index)

        coverage = (
            len(matched)
            / total
        )

        result.append({
            "k": k,
            "coverage": coverage,
            "matched": len(matched),
            "total": total
        })

    return result


# ============================================================
# ТАБЛИЦА COVERAGE
# ============================================================

def get_coverage_at(
    curve,
    k
):

    if not curve:
        return 0.0

    k = min(k, len(curve))

    return curve[k - 1]["coverage"]


def first_k_for_coverage(
    curve,
    target
):

    for item in curve:

        if item["coverage"] >= target:
            return item["k"]

    return None


# ============================================================
# КАЧЕСТВО ПАТТЕРНОВ
# ============================================================

def pattern_statistics(
    train_counter,
    train_instances,
    valid_instances,
    test_instances,
    key
):

    patterns = [
        pattern
        for pattern, _ in
        train_counter.most_common()
    ]

    train_curve = cumulative_coverage(
        patterns,
        train_instances,
        key
    )

    valid_curve = cumulative_coverage(
        patterns,
        valid_instances,
        key
    )

    test_curve = cumulative_coverage(
        patterns,
        test_instances,
        key
    )

    return {
        "patterns": patterns,
        "train_curve": train_curve,
        "validation_curve": valid_curve,
        "test_curve": test_curve
    }


# ============================================================
# DOCUMENT COVERAGE
# ============================================================

def document_coverage(
    patterns,
    documents,
    key
):
    """
    Сколько документов имеют хотя бы одну ссылку,
    которую покрывает набор паттернов.

    Это полезно для оценки обобщения именно между
    документами.
    """

    if not documents:
        return 0.0

    covered = 0

    for document in documents:

        found = False

        for reference in document["references"]:

            text = reference["text"]

            if key == "formal":
                signature = formal_tokens(text)

            elif key == "shape":
                signature = shape_tokens(text)

            else:
                signature = semantic_signature(
                    reference
                )

            if signature in patterns:
                found = True
                break

        if found:
            covered += 1

    return covered / len(documents)


# ============================================================
# УНИКАЛЬНЫЕ ПАТТЕРНЫ
# ============================================================

def pattern_diversity(
    counter,
    total
):

    unique = len(counter)

    singleton = sum(
        1
        for count in counter.values()
        if count == 1
    )

    if total == 0:
        singleton_ratio = 0
    else:
        singleton_ratio = singleton / total

    return {
        "unique_patterns": unique,
        "singleton_patterns": singleton,
        "singleton_ratio": singleton_ratio
    }


# ============================================================
# ENTROPY
# ============================================================

def entropy(counter):

    total = sum(counter.values())

    if total == 0:
        return 0.0

    value = 0.0

    for count in counter.values():

        p = count / total

        value -= p * math.log2(p)

    return value


# ============================================================
# PRINT
# ============================================================

def print_curve(
    name,
    curve
):

    print()
    print(name)
    print("-" * 70)

    if not curve:
        print("Нет данных")
        return

    checkpoints = [
        1,
        2,
        5,
        10,
        20,
        30,
        50,
        75,
        100
    ]

    for k in checkpoints:

        if k > len(curve):
            continue

        item = curve[k - 1]

        print(
            f"  top-{k:3d}: "
            f"{item['coverage'] * 100:6.2f}% "
            f"({item['matched']}/{item['total']})"
        )


def print_thresholds(
    name,
    curve
):

    print()
    print(name)
    print("-" * 70)

    for target in [
        0.50,
        0.70,
        0.80,
        0.90,
        0.95,
        0.99
    ]:

        k = first_k_for_coverage(
            curve,
            target
        )

        if k is None:
            print(
                f"  {target * 100:5.1f}% : "
                f"НЕ ДОСТИГНУТО"
            )
        else:
            print(
                f"  {target * 100:5.1f}% : "
                f"{k} паттернов"
            )


def print_top_patterns(
    counter,
    name,
    limit=20
):

    print()
    print(name)
    print("-" * 70)

    total = sum(counter.values())

    for index, (
        pattern,
        count
    ) in enumerate(
        counter.most_common(limit),
        start=1
    ):

        percentage = (
            count / total * 100
            if total
            else 0
        )

        print(
            f"{index:3d}. "
            f"{count:5d} "
            f"({percentage:6.2f}%) "
            f"{' '.join(pattern)}"
        )


# ============================================================
# СЕРИАЛИЗАЦИЯ ПАТТЕРНОВ
# ============================================================

def serialize_pattern(pattern):
    return list(pattern)


def serialize_counter(counter):

    result = []

    total = sum(counter.values())

    for pattern, count in counter.most_common():

        result.append({
            "pattern":
                serialize_pattern(pattern),

            "frequency":
                count,

            "support":
                (
                    count / total
                    if total
                    else 0
                )
        })

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Эксперимент по проверке "
            "обобщаемости формальных "
            "паттернов библиографических ссылок."
        )
    )

    parser.add_argument(
        "data",
        help="Каталог DataSource"
    )

    parser.add_argument(
        "--output",
        default="localization_patterns.json",
        help="Файл результатов"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    data_dir = Path(args.data)

    if not data_dir.exists():
        raise FileNotFoundError(
            data_dir
        )

    # ========================================================
    # LOAD
    # ========================================================

    print("=" * 80)
    print("PATTERN MINER")
    print("=" * 80)

    documents = load_dataset(
        data_dir
    )

    print()
    print(
        f"Документов загружено: "
        f"{len(documents)}"
    )

    print(
        f"Ссылок: "
        f"{sum(len(d['references']) for d in documents)}"
    )

    if len(documents) < 3:
        raise RuntimeError(
            "Слишком мало документов для train/test."
        )

    # ========================================================
    # SPLIT
    # ========================================================

    split = make_split(
        documents,
        seed=args.seed
    )

    split_documents_data = split_documents(
        documents,
        split
    )

    print()
    print("=" * 80)
    print("DATASET SPLIT")
    print("=" * 80)

    for subset in [
        "train",
        "validation",
        "test"
    ]:

        docs = split_documents_data[subset]

        refs = sum(
            len(d["references"])
            for d in docs
        )

        print(
            f"{subset:12s}: "
            f"{len(docs):3d} docs, "
            f"{refs:4d} references"
        )

    # ========================================================
    # INSTANCES
    # ========================================================

    train_instances = make_instances(
        split_documents_data["train"]
    )

    valid_instances = make_instances(
        split_documents_data["validation"]
    )

    test_instances = make_instances(
        split_documents_data["test"]
    )

    # ========================================================
    # FORMAL
    # ========================================================

    train_formal = count_patterns(
        train_instances,
        "formal"
    )

    train_shape = count_patterns(
        train_instances,
        "shape"
    )

    # ========================================================
    # SEMANTIC UPPER BOUND
    # ========================================================

    train_semantic = count_patterns(
        train_instances,
        "semantic"
    )

    # ========================================================
    # BASIC STATISTICS
    # ========================================================

    print()
    print("=" * 80)
    print("DIVERSITY")
    print("=" * 80)

    for name, counter in [
        ("FORMAL", train_formal),
        ("SHAPE", train_shape),
        ("SEMANTIC", train_semantic)
    ]:

        stats = pattern_diversity(
            counter,
            len(train_instances)
        )

        print()
        print(name)

        print(
            f"  unique:       "
            f"{stats['unique_patterns']}"
        )

        print(
            f"  singletons:   "
            f"{stats['singleton_patterns']}"
        )

        print(
            f"  singleton %:  "
            f"{stats['singleton_ratio'] * 100:.2f}%"
        )

        print(
            f"  entropy:      "
            f"{entropy(counter):.3f} bits"
        )

    # ========================================================
    # TOP PATTERNS
    # ========================================================

    print_top_patterns(
        train_formal,
        "TOP FORMAL PATTERNS",
        20
    )

    print_top_patterns(
        train_shape,
        "TOP GENERALIZED PATTERNS",
        20
    )

    print_top_patterns(
        train_semantic,
        "TOP SEMANTIC STRUCTURES",
        20
    )

    # ========================================================
    # COVERAGE
    # ========================================================

    formal_stats = pattern_statistics(
        train_formal,
        train_instances,
        valid_instances,
        test_instances,
        "formal"
    )

    shape_stats = pattern_statistics(
        train_shape,
        train_instances,
        valid_instances,
        test_instances,
        "shape"
    )

    semantic_stats = pattern_statistics(
        train_semantic,
        train_instances,
        valid_instances,
        test_instances,
        "semantic"
    )

    # ========================================================
    # CURVES
    # ========================================================

    print()
    print("=" * 80)
    print("FORMAL PATTERN COVERAGE")
    print("=" * 80)

    print_curve(
        "TRAIN",
        formal_stats["train_curve"]
    )

    print_curve(
        "VALIDATION",
        formal_stats["validation_curve"]
    )

    print_curve(
        "TEST",
        formal_stats["test_curve"]
    )

    print_thresholds(
        "FORMAL — TRAIN",
        formal_stats["train_curve"]
    )

    print_thresholds(
        "FORMAL — VALIDATION",
        formal_stats["validation_curve"]
    )

    print_thresholds(
        "FORMAL — TEST",
        formal_stats["test_curve"]
    )

    # ========================================================
    # GENERALIZED
    # ========================================================

    print()
    print("=" * 80)
    print("GENERALIZED PATTERN COVERAGE")
    print("=" * 80)

    print_curve(
        "TRAIN",
        shape_stats["train_curve"]
    )

    print_curve(
        "VALIDATION",
        shape_stats["validation_curve"]
    )

    print_curve(
        "TEST",
        shape_stats["test_curve"]
    )

    print_thresholds(
        "GENERALIZED — TRAIN",
        shape_stats["train_curve"]
    )

    print_thresholds(
        "GENERALIZED — VALIDATION",
        shape_stats["validation_curve"]
    )

    print_thresholds(
        "GENERALIZED — TEST",
        shape_stats["test_curve"]
    )

    # ========================================================
    # SEMANTIC UPPER BOUND
    # ========================================================

    print()
    print("=" * 80)
    print("SEMANTIC STRUCTURE — UPPER BOUND")
    print("=" * 80)

    print_curve(
        "TRAIN",
        semantic_stats["train_curve"]
    )

    print_curve(
        "VALIDATION",
        semantic_stats["validation_curve"]
    )

    print_curve(
        "TEST",
        semantic_stats["test_curve"]
    )

    print_thresholds(
        "SEMANTIC — TRAIN",
        semantic_stats["train_curve"]
    )

    print_thresholds(
        "SEMANTIC — VALIDATION",
        semantic_stats["validation_curve"]
    )

    print_thresholds(
        "SEMANTIC — TEST",
        semantic_stats["test_curve"]
    )

    # ========================================================
    # DOCUMENT GENERALIZATION
    # ========================================================

    print()
    print("=" * 80)
    print("DOCUMENT GENERALIZATION")
    print("=" * 80)

    for name, counter, key in [
        (
            "FORMAL",
            train_formal,
            "formal"
        ),
        (
            "SHAPE",
            train_shape,
            "shape"
        ),
        (
            "SEMANTIC",
            train_semantic,
            "semantic"
        )
    ]:

        patterns = set(counter.keys())

        valid_doc_cov = document_coverage(
            patterns,
            split_documents_data["validation"],
            key
        )

        test_doc_cov = document_coverage(
            patterns,
            split_documents_data["test"],
            key
        )

        print()
        print(name)

        print(
            f"  validation documents: "
            f"{valid_doc_cov * 100:.2f}%"
        )

        print(
            f"  test documents:       "
            f"{test_doc_cov * 100:.2f}%"
        )

    # ========================================================
    # UNSEEN PATTERN RATE
    # ========================================================

    print()
    print("=" * 80)
    print("UNSEEN PATTERNS")
    print("=" * 80)

    for name, train_counter, key in [
        (
            "FORMAL",
            train_formal,
            "formal"
        ),
        (
            "SHAPE",
            train_shape,
            "shape"
        ),
        (
            "SEMANTIC",
            train_semantic,
            "semantic"
        )
    ]:

        known = set(
            train_counter.keys()
        )

        print()
        print(name)

        for subset_name, instances in [
            ("validation", valid_instances),
            ("test", test_instances)
        ]:

            unseen = sum(
                1
                for item in instances
                if item[key] not in known
            )

            ratio = (
                unseen / len(instances)
                if instances
                else 0
            )

            print(
                f"  {subset_name:12s}: "
                f"{ratio * 100:6.2f}% "
                f"({unseen}/{len(instances)})"
            )

    # ========================================================
    # СОХРАНЕНИЕ
    # ========================================================

    output = {
        "experiment": {
            "seed": args.seed,
            "reference_label": REFERENCE_LABEL,

            "split": split,

            "description": (
                "Formal pattern generalization "
                "experiment. Patterns are mined "
                "only on training documents and "
                "evaluated on held-out documents."
            )
        },

        "dataset": {
            "documents":
                len(documents),

            "references":
                len(train_instances)
                + len(valid_instances)
                + len(test_instances),

            "train_references":
                len(train_instances),

            "validation_references":
                len(valid_instances),

            "test_references":
                len(test_instances)
        },

        "formal": {
            "statistics":
                pattern_diversity(
                    train_formal,
                    len(train_instances)
                ),

            "patterns":
                serialize_counter(
                    train_formal
                ),

            "coverage": {
                "train":
                    formal_stats["train_curve"],

                "validation":
                    formal_stats["validation_curve"],

                "test":
                    formal_stats["test_curve"]
            }
        },

        "generalized": {
            "statistics":
                pattern_diversity(
                    train_shape,
                    len(train_instances)
                ),

            "patterns":
                serialize_counter(
                    train_shape
                ),

            "coverage": {
                "train":
                    shape_stats["train_curve"],

                "validation":
                    shape_stats["validation_curve"],

                "test":
                    shape_stats["test_curve"]
            }
        },

        "semantic_upper_bound": {
            "statistics":
                pattern_diversity(
                    train_semantic,
                    len(train_instances)
                ),

            "patterns":
                serialize_counter(
                    train_semantic
                ),

            "coverage": {
                "train":
                    semantic_stats["train_curve"],

                "validation":
                    semantic_stats["validation_curve"],

                "test":
                    semantic_stats["test_curve"]
            }
        }
    }

    with open(
        args.output,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("=" * 80)
    print(
        f"Результат сохранён: {args.output}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()