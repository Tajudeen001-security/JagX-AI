# JagX Multilingual + African Language Layer

JagX treats language as a routing capability rather than a hard-coded UI language.

## Goals
- Detect the language from text or speech.
- Keep the user's language as the response language unless the user requests another language.
- Support text input/output, speech recognition (ASR), and speech synthesis (TTS) through pluggable providers.
- Maintain an extensible registry for African languages and dialects, including low-resource languages.
- Permit code, websites, games, documents, and other generated artifacts to be produced in the requested language where the target toolchain supports it.

## Important implementation note
The registry does not magically make a base model fluent in 2,000+ languages. Actual quality requires language-specific text/audio data, evaluation sets, tokenization coverage, and ASR/TTS models. The training pipeline can progressively add these resources without changing the agent interface.

See `registry.py` and `language_router.py` for the provider-neutral interfaces.
