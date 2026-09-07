# JagX African AI program

JagX is being extended toward a Pan-African language and knowledge system. The target is **about 400 African languages**, with support added progressively as verified resources become available.

## Interaction model

For each supported language, JagX is designed to support:

- written input and written responses;
- spoken input through an ASR adapter;
- spoken responses through a TTS adapter;
- automatic language detection from text or speech;
- keeping the response in the user's language unless the user asks to switch;
- dialect metadata where reliable data exists.

The voice layer is an adapter contract rather than a hard dependency on an external AI provider. This keeps the core JagX model provider-independent.

## Data strategy

The first training run must not pretend that 400 languages have equal resources. A language is promoted to a supported tier only when we have:

1. a verified language identifier and script;
2. legally reusable or permissioned text data;
3. speech data where voice support is claimed;
4. language/dialect metadata;
5. quality checks and deduplication;
6. community or expert validation where practical.

The repository now contains `data/africa/language_registry.json` with a 400-language target and an initial priority set. The manifest is intentionally conservative: it does not fabricate missing corpora.

## Local problem-solving domains

African-language data should be balanced with locally relevant material covering:

- agriculture and livestock;
- public-health education and health navigation (not diagnosis claims);
- schools, literacy and teacher support;
- roads, water, electricity and infrastructure;
- public services and civic information;
- small businesses, markets and local economic activity;
- climate, disaster preparedness and environmental knowledge;
- African history, culture and oral heritage.

Domain data should preserve provenance and licensing. High-stakes health and public-service answers require explicit safety policies and should distinguish general information from professional advice.

## Why this matters

### Representation and bias

Increasing high-quality African-language data gives evaluation and training a better representation of African users rather than relying mainly on English/French/other high-resource-language proxies.

### Digital sovereignty

The long-term design keeps the core model, tokenizer, training pipeline and inference stack under JagX control. External datasets and adapters can be used where their licences permit, but the architecture must not require a foreign hosted model for basic operation.

### Research foundation

Masakhane projects demonstrate the importance of community-centered African NLP, including text, NER and speech work. NLLB provides useful multilingual data and language-code conventions, but its coverage must not be mistaken for complete coverage of Africa. New languages therefore need their own data-collection and validation path.

## Training phases

**Phase A — foundation:** keep the existing JagX pretraining pipeline stable.

**Phase B — multilingual tokenizer/data:** add verified African text to a multilingual corpus and measure per-language tokenization quality.

**Phase C — instruction tuning:** add language-specific question/answer, translation, reasoning and local-domain examples with provenance.

**Phase D — speech:** train/evaluate ASR and TTS adapters for languages with enough speech data; do not mark unsupported languages as voice-capable.

**Phase E — evaluation:** maintain per-language benchmarks for comprehension, generation, translation, safety, hallucination, dialect robustness and local-domain usefulness.

**Phase F — expansion:** grow the registry toward 400 languages through community datasets and validated open resources.
