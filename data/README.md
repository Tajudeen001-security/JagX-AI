# JagX Data Pipeline

Training data is a first-class subsystem. Each dataset should retain provenance, license/source metadata, language, domain, quality score, processing version and contamination status.

Pipeline: ingest → normalize → filter → deduplicate → quality-score → split → tokenize → pack → manifest.

Do not commit large raw datasets or model weights to Git. Store manifests and reproducible processing code in this repository; use replaceable object storage for large artifacts.
