# JagX Data Pipeline

Training data is a first-class subsystem. Each dataset should retain provenance, license/source metadata, language, domain, quality score, processing version and contamination status.

Pipeline: ingest → normalize → filter → deduplicate → quality-score → split → tokenize → pack → manifest.

## Real training corpora

JagX now has a reproducible corpus registry in `data/manifests/open_corpora.yaml` and a downloader at `scripts/download_training_data.py`.

The initial real sources are:

- **OpenAssistant/oasst1** — instruction/conversation data, Apache-2.0.
- **FineWeb2** — multilingual web pretraining data, ODC-By.
- **Dolma** — large mixed pretraining corpus, ODC-By.
- **Common Crawl** — optional web-scale source; its terms require respecting the rights and terms applying to the crawled material.

For a first training experiment, download a bounded OASST1 slice locally:

```bash
python scripts/download_training_data.py --source oasst1 --max-rows 50000
```

Then run the existing normalization/filtering/tokenization pipeline against `data/raw/`.

### Why the raw corpus is not committed

Large raw corpora should not be copied into the Git repository. OASST1 alone has an 84.4k-row training split, while serious pretraining corpora are measured in billions or trillions of tokens. The repository therefore keeps the **real corpus sources, licenses, revisions, downloader, and provenance manifests**, while the actual multi-gigabyte artifacts live outside Git. This keeps the repo usable from a phone and makes the data reproducible instead of hiding an undocumented binary dump.

Before commercial use, verify the current license/terms of every upstream source and preserve its provenance/removal requirements.
