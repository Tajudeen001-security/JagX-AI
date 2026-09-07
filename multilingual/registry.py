from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageSpec:
    code: str
    name: str
    region: str
    text: bool = True
    speech_in: bool = False
    speech_out: bool = False
    low_resource: bool = False


# Seed registry. More languages are data, not application logic: append validated
# ISO 639-3 records as training/evaluation resources become available.
AFRICAN_LANGUAGE_SEEDS: tuple[LanguageSpec, ...] = (
    LanguageSpec("yor", "Yoruba", "West Africa", speech_in=True, speech_out=True),
    LanguageSpec("hau", "Hausa", "West Africa", speech_in=True, speech_out=True),
    LanguageSpec("ibo", "Igbo", "West Africa", speech_in=True, speech_out=True),
    LanguageSpec("pcm", "Nigerian Pidgin", "West Africa", speech_in=True, speech_out=True),
    LanguageSpec("swa", "Swahili", "East Africa", speech_in=True, speech_out=True),
    LanguageSpec("amh", "Amharic", "East Africa", speech_in=True, speech_out=True),
    LanguageSpec("som", "Somali", "Horn of Africa", speech_in=True, speech_out=True),
    LanguageSpec("tir", "Tigrinya", "Horn of Africa", speech_in=True, speech_out=True),
    LanguageSpec("orm", "Oromo", "East Africa", speech_in=True, speech_out=True),
    LanguageSpec("zul", "Zulu", "Southern Africa", speech_in=True, speech_out=True),
    LanguageSpec("xho", "Xhosa", "Southern Africa", speech_in=True, speech_out=True),
    LanguageSpec("afr", "Afrikaans", "Southern Africa", speech_in=True, speech_out=True),
    LanguageSpec("sna", "Shona", "Southern Africa", speech_in=True, speech_out=True),
    LanguageSpec("kin", "Kinyarwanda", "East Africa", speech_in=True, speech_out=True),
    LanguageSpec("lug", "Ganda", "East Africa", speech_in=True, speech_out=True),
    LanguageSpec("wol", "Wolof", "West Africa", speech_in=True, speech_out=True),
    LanguageSpec("ful", "Fula", "West Africa", low_resource=True),
    LanguageSpec("bam", "Bambara", "West Africa", speech_in=True, speech_out=True),
    LanguageSpec("twi", "Twi", "West Africa", speech_in=True, speech_out=True),
    LanguageSpec("gaa", "Ga", "West Africa", low_resource=True),
    LanguageSpec("fon", "Fon", "West Africa", low_resource=True),
    LanguageSpec("ewe", "Ewe", "West Africa", speech_in=True, speech_out=True),
    LanguageSpec("lin", "Lingala", "Central Africa", speech_in=True, speech_out=True),
    LanguageSpec("kon", "Kongo", "Central Africa", low_resource=True),
    LanguageSpec("lua", "Luba-Lulua", "Central Africa", low_resource=True),
    LanguageSpec("mos", "Mossi", "West Africa", low_resource=True),
    LanguageSpec("fat", "Fanti", "West Africa", low_resource=True),
    LanguageSpec("dag", "Dagbani", "West Africa", low_resource=True),
    LanguageSpec("dyu", "Dyula", "West Africa", low_resource=True),
    LanguageSpec("tem", "Temne", "West Africa", low_resource=True),
    LanguageSpec("men", "Mende", "West Africa", low_resource=True),
    LanguageSpec("kab", "Kabyle", "North Africa", low_resource=True),
    LanguageSpec("ber", "Berber", "North Africa", low_resource=True),
    LanguageSpec("egy", "Egyptian Arabic", "North Africa", speech_in=True, speech_out=True),
    LanguageSpec("ary", "Moroccan Arabic", "North Africa", speech_in=True, speech_out=True),
)


def get_seed_registry() -> dict[str, LanguageSpec]:
    return {item.code: item for item in AFRICAN_LANGUAGE_SEEDS}


def merge_registry(*registries: dict[str, LanguageSpec]) -> dict[str, LanguageSpec]:
    merged: dict[str, LanguageSpec] = {}
    for registry in registries:
        merged.update(registry)
    return merged
