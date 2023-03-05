import spacy
from spacy import displacy
from spacy.tokens import Doc, Token

import hjson

from typing import Mapping, MutableMapping, Optional, Sequence, Iterable

EntityMetadata = Mapping[str, str]
DBSchemeMetadata = Mapping[str, EntityMetadata]

nlp = spacy.load("uk_core_news_lg")

# Алгоритм: пройтись по тексту, распознать все сущности из схемы базы, после чего
# пройтись по предложению и найти все поля, которые могут быть связаны с распознанными сущностями

def similarity_func(a: str, b: str) -> float:
    return nlp(a).similarity(nlp(b))

def find_entity_key(scheme: DBSchemeMetadata, entity: str, min_similarity: float) -> tuple[Optional[str], Optional[float]]:
    similar_keys = [key for key in scheme if similarity_func(key, entity) > min_similarity]
    if similar_keys:
        most_similar = max(similar_keys, key=lambda key: similarity_func(key, entity))
        return most_similar, similarity_func(most_similar, entity)

    return None, None

def get_max_entity_name_length(scheme: DBSchemeMetadata) -> int:
    """Возвращает максимальную длину (в словах) имени сущности в схеме"""
    return max(len(key.split()) for key in scheme)

def untokenize(tokens: Sequence[Token]) -> str:
    """Собирает токены в текст."""
    return ''.join([token.text_with_ws for token in tokens])

def get_entity_pretendents(scheme: DBSchemeMetadata, doc: Doc) -> Sequence[Sequence[Token]]:
    """Возвращает набор строк, которые могут быть сущностями в схеме"""

    max_entity_name_length = get_max_entity_name_length(scheme)

    entity_pretendents = []
    # Проходимся линейно по тексту и собираем все возможные
    # куски предложения длиной от 1 до max_entity_name_length
    for i in range(len(doc)):
        for j in range(0, max_entity_name_length):
            if i + j >= len(doc):
                break
            entity_pretendents.append(doc[i:i+j+1])

    return entity_pretendents

def find_entities(scheme: DBSchemeMetadata, doc: Doc, min_similarity: float) -> Sequence[str]:
    """Возвращает список сущностей в тексте, которые есть в схеме базы данных"""
    entity_pretendents = get_entity_pretendents(scheme, doc)

    # Проходимся по всем возможным сущностям и ищем их в схеме базы данных
    entities = []
    for entity_pretendent in entity_pretendents:
        entity_name = untokenize(entity_pretendent)
        entity_key, similarity = find_entity_key(scheme, entity_name, min_similarity)
        if entity_key:
            print(f"Found entity: {entity_name} = {entity_key} (similarity: {similarity})")
            entities.append(entity_key)

    return entities

def analyse(scheme: DBSchemeMetadata, item: Mapping[str, str], min_similarity: float):
    find_entities(scheme, nlp(item["text"]), min_similarity)

with open("data.hjson", "r", encoding="utf-8") as f:
    data = hjson.load(f)

with open("scheme.hjson", "r", encoding="utf-8") as f:
    scheme: DBSchemeMetadata = hjson.load(f)

for item in data:
    analyse(scheme, item, 0.6)
