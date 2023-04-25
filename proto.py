import spacy
from spacy import displacy
from spacy.tokens import Doc, Token
from thefuzz import fuzz
import matplotlib.pyplot as plt

import hjson

from operator import itemgetter
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Iterable

EntityMetadata = Mapping[str, str]
DBSchemeMetadata = Mapping[str, EntityMetadata]
AliasDictionary = Mapping[str, str]

nlp = spacy.load("uk_core_news_lg")

# Алгоритм: пройтись по тексту, распознать все сущности из схемы базы, после чего
# пройтись по предложению и найти все поля, которые могут быть связаны с распознанными сущностями

lookup_registry: dict = {}

def lookup(key: str) -> Doc:
    if key not in lookup_registry:
        r = lookup_registry[key] = nlp(key)
        return r
    return lookup_registry[key]

def similarity_func(a: str, b: str) -> float:
    return lookup(a).similarity(lookup(b))

# def similarity_func(a: str, b: str) -> float:
#     return fuzz.ratio(a, b) / 100

def find_entity_key(scheme: DBSchemeMetadata, aliases: AliasDictionary, entity: str, min_similarity: float) -> tuple[Optional[str], Optional[float]]:
    # similar_aliased = [(aliases[alias], similarity_func(alias, entity)) for alias in aliases]
    similar_keys = [(key, similarity_func(key, entity)) for key in scheme]

    most_similar, similarity_value = max(similar_keys, key=itemgetter(1))

    if similarity_value >= min_similarity:
        return most_similar, similarity_value

    return None, None

def get_max_entity_name_length(scheme: DBSchemeMetadata, aliases: AliasDictionary) -> int:
    """Возвращает максимальную длину (в словах) имени сущности в схеме"""
    return max(len(key.split()) for key in scheme)

def untokenize(tokens: Sequence[Token]) -> str:
    """Собирает токены в текст."""
    return ''.join([token.text_with_ws for token in tokens])

def get_entity_pretendents(scheme: DBSchemeMetadata, aliases: AliasDictionary, doc: Doc) -> Sequence[Sequence[Token]]:
    """Возвращает набор строк, которые могут быть сущностями в схеме"""

    max_entity_name_length = get_max_entity_name_length(scheme, aliases)

    entity_pretendents = []
    # Проходимся линейно по тексту и собираем все возможные
    # куски предложения длиной от 1 до max_entity_name_length
    for i in range(len(doc)):
        for j in range(0, max_entity_name_length):
            if i + j >= len(doc):
                break
            entity_pretendents.append(doc[i:i+j+1])

    return entity_pretendents

def find_entities(scheme: DBSchemeMetadata, aliases: AliasDictionary, doc: Doc, min_similarity: float) -> Sequence[str]:
    """Возвращает список сущностей в тексте, которые есть в схеме базы данных"""
    entity_pretendents = get_entity_pretendents(scheme, aliases, doc)

    # Проходимся по всем возможным сущностям и ищем их в схеме базы данных
    entities = []
    for entity_pretendent in entity_pretendents:
        entity_name = untokenize(entity_pretendent)
        entity_key, similarity = find_entity_key(scheme, aliases, entity_name, min_similarity)
        if entity_key:
            # print(f"Found entity: {entity_name} = {entity_key} (similarity: {similarity})")
            entities.append(entity_key)

    return entities

def analyse_confusion_matrix(scheme: DBSchemeMetadata, aliases: AliasDictionary, item: Mapping[str, Any], min_similarity: float):
    db_entities_total = len(scheme)
    actual_entities = set(item["expected_entities"])
    predicted_entities = set(find_entities(scheme, aliases, nlp(item["text"]), min_similarity))

    tp = len(actual_entities & predicted_entities)
    fp = len(predicted_entities - actual_entities)
    fn = len(actual_entities - predicted_entities)
    tn = db_entities_total - tp - fp - fn

    return tp, fp, fn, tn

def analyse_all(scheme: DBSchemeMetadata, aliases: AliasDictionary, data: Sequence[Mapping[str, Any]], threshold: float):
    tp, fp, fn, tn = 0, 0, 0, 0
    for i, item in enumerate(data):
        print(f"Processing item {i+1}/{len(data)}...")
        tp_, fp_, fn_, tn_ = analyse_confusion_matrix(scheme, aliases, item, threshold)
        tp += tp_
        fp += fp_
        fn += fn_
        tn += tn_

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    return accuracy, precision, recall, f1

def plot_metrics_by_threshold(scheme: DBSchemeMetadata, aliases: AliasDictionary,
                              data: Sequence[Mapping[str, Any]], resolution: int):
    thresholds = [i / resolution for i in range(0, resolution + 1)]
    metrics = []
    for threshold in thresholds:
        print(f"Processing threshold {threshold}...")
        accuracy, precision, recall, f1 = analyse_all(scheme, aliases, data, threshold)
        metrics.append((accuracy, precision, recall, f1))

    metrics = list(zip(*metrics))

    plt.step(thresholds, metrics[0], drawstyle='steps-mid', label="Accuracy")
    plt.step(thresholds, metrics[1], drawstyle='steps-mid', label="Precision")
    plt.step(thresholds, metrics[2], drawstyle='steps-mid', label="Recall")
    plt.step(thresholds, metrics[3], drawstyle='steps-mid', label="F1")

    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.legend()

    plt.show()

with open("data.hjson", "r", encoding="utf-8") as f:
    data: Sequence[Mapping[str, Any]] = hjson.load(f) # type: ignore

with open("scheme.hjson", "r", encoding="utf-8") as f:
    scheme: DBSchemeMetadata = hjson.load(f)

with open("scheme_dict.hjson", "r", encoding="utf-8") as f:
    aliases: AliasDictionary = hjson.load(f)

plot_metrics_by_threshold(scheme, aliases, data, resolution=100)