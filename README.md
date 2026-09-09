# JagX AI

Independent AI research platform. Core model, tokenizer, training, inference, agent, sandbox, coding loop and evaluation run **without** another AI provider API.

**No demo mode.** Unbound generation returns an error until a real checkpoint is trained and loaded.

Scale configs now go from a 10M smoke model to ~3B (`configs/kaggle.json` through `configs/xlarge.json`). Bigger JSON is not a trained brain — run `jagx inspect` then train.

## Status

| Area | State |
|------|-------|
| Transformer (RoPE, RMSNorm, SwiGLU, GQA, KV-cache, SDPA) | Real + tested |
| Tokenizer BPE | Real + tested |
| Training + checkpoint resume | Real + tested |
| Scale ladder (tiny → xlarge) + `jagx inspect` | Real + tested |
| Inference from checkpoint | Real (requires your weights) |
| Unified orchestrator | Real + tested |
| Agent TaskDAG | Real + tested |
| Sandbox + coding write→test | Real + tested |
| Memory | Real + tested |
| Task experience learning store | Real + tested |
| Bounded computer-control contract | Real + tested; requires local host adapter |
| Vision | Trainable image encoder + perception contracts |
| Multilingual routing | Real routing contract; language quality depends on trained data/models |
| Vehicle autonomy | Safety-first simulation/planning contracts; **not road-ready** |
| Gaming instruction seed corpus | Real MIT seed + collector |
| Local API | Real; generate needs checkpoint |
| Paper trading | Real paper-only |
| Media / multimodal quality | Architecture present; needs trained media weights |

**Not claimed:** frontier chat quality, universal African-language fluency, or production autonomous driving without the required trained models, evaluation and safety validation.

## New agent architecture

JagX now separates the model from the capabilities around it:

- `agent/computer_control.py` — explicit user-granted screen, cursor, keyboard, terminal and filesystem control contracts.
- `memory/learning.py` — persistent task experience that can be retrieved without retraining model weights after every task.
- `multimodal/perception.py` — camera/screen/video perception adapter contracts.
- `multilingual/router.py` — automatic response-language routing with English fallback.
- `autonomy/vehicle.py` — perception-to-planning interfaces and an independent safety envelope for future simulation/robotics work.

Computer access is deny-by-default at the integration boundary. `FULL_CONTROL` is available only when the host application explicitly grants it, and destructive operations still request confirmation by default.

# Complete Kaggle GPU Training Guide

This is the main procedure for training JagX-AI on a Kaggle GPU from your phone. You do **not** need Android Studio, a local GPU, or a powerful Windows PC for this training path.

The repository already contains the Kaggle notebook and launcher. The notebook clones the current `main` branch, installs JagX and the Hugging Face dataset client, verifies CUDA, downloads the public OpenAssistant/oasst1 corpus, prepares deterministic train/validation data, trains the BPE tokenizer, checks that real batches exist, and then starts native CUDA training. fileciteturn2file0

> **Important:** this is real model training, but the default Kaggle run is a bounded research run. It is **not** a 200B-parameter training run. A 200B model requires a large distributed GPU cluster and a much larger data/training budget. We will scale JagX gradually after the pipeline is proven.

## 1. What you need

You need:

1. A GitHub account with access to `Tajudeen001-security/JagX-AI`.
2. A Kaggle account.
3. A Kaggle Notebook.
4. Kaggle **Internet = ON**.
5. Kaggle **Accelerator = GPU**.
6. The repository's `notebooks/jagx_kaggle_train.ipynb` notebook.

The current notebook is designed to clone the repository automatically, so you do not need to manually upload the whole source tree. fileciteturn3file0

## 2. Open the Kaggle notebook

Open Kaggle and create a new Notebook.

In the Notebook settings:

- Turn **Internet** on.
- Set **Accelerator** to **GPU**.
- Use the available NVIDIA GPU Kaggle gives your session, such as T4 or P100 when available.

Then import/open:

`notebooks/jagx_kaggle_train.ipynb`

The notebook's first code cell clones `https://github.com/Tajudeen001-security/JagX-AI.git` into `/kaggle/working/JagX-AI` and resets it to the current `origin/main`. fileciteturn3file0

## 3. Run the notebook from top to bottom

Run the cells in this exact order.

### Cell 1 — clone JagX-AI

You normally do not need to change anything.

It downloads the current `main` branch into Kaggle and prints the commit being trained.

### Cell 2 — install dependencies and verify GPU

The notebook runs:

```bash
pip install -q -e . datasets huggingface_hub
```

Then it checks:

```python
import torch
print(torch.cuda.is_available())
```

You need:

```text
CUDA available: True
```

It will also print the NVIDIA GPU name.

If it says `False`, **stop**. Do not start training. Go back to Kaggle Settings, select a GPU accelerator, and restart the session. The notebook intentionally fails early instead of allowing a CPU training run. fileciteturn3file0

### Cell 3 — start training

The notebook currently sets these safe starting values:

```python
JAGX_ROWS = 50000
JAGX_STEPS = 1000
JAGX_SEQ_LEN = 512
JAGX_BATCH_SIZE = 4
JAGX_GRAD_ACCUM = 8
```

Then it runs:

```bash
python scripts/kaggle_train.py --skip-pip
```

This is the first training run you should complete successfully before increasing anything. fileciteturn3file0

### Cell 4 — inspect the checkpoint

The final notebook cell checks `kaggle_checkpoints/` and prints the generated `.pt` checkpoint files and their sizes. fileciteturn3file0

## 4. What the training pipeline actually does

The training process is not simply "download data and press train". The launcher performs several checks first.

### Step A — download training data

The current pipeline uses OpenAssistant/oasst1 through the Hugging Face `datasets` ecosystem.

The data is kept in Kaggle working storage rather than committed to GitHub.

### Step B — prepare the corpus

The pipeline creates deterministic training and validation data and removes exact duplicates.

### Step C — train the tokenizer

JagX trains its BPE tokenizer from the prepared corpus.

### Step D — verify a real batch exists

The launcher checks that the prepared corpus can actually produce a fixed-size training batch before the trainer starts.

This is important because the earlier `training batches are empty` class of failure should be caught before expensive training begins.

### Step E — train on CUDA

Once the checks pass, JagX starts native PyTorch CUDA training using the configured Transformer.

### Step F — save checkpoints

Checkpoints are written to Kaggle working storage under:

```text
kaggle_checkpoints/
```

Do **not** commit checkpoints or large generated datasets to this GitHub repository. The source code and reproducible preparation logic belong in GitHub; large training artifacts belong in external storage. fileciteturn2file0

## 5. The first run: do not increase the model yet

Your first goal is simple:

**Get one complete GPU training run to finish successfully.**

Do not immediately jump to billions of parameters.

After the first successful run, we inspect:

- training loss
- validation loss
- tokens processed
- GPU memory usage
- checkpoint size
- training speed
- whether loss is actually decreasing
- whether the checkpoint can be loaded for inference

Only after those checks pass should we increase model size and training data.

## 6. Increase training gradually

The Kaggle training documentation currently describes a larger bounded configuration around:

- 80,000 OASST1 records
- 5% validation
- sequence length 512
- batch size 4
- gradient accumulation 8
- 8 Transformer layers
- hidden size 512
- 8 attention heads
- 4 KV heads
- AdamW
- learning rate `3e-4`
- 3,000 steps

That configuration is still a research-scale run, not frontier training. fileciteturn2file0

You can change the run with environment variables, for example:

```python
import os
os.environ['JAGX_ROWS'] = '80000'
os.environ['JAGX_STEPS'] = '3000'
os.environ['JAGX_SEQ_LEN'] = '512'
os.environ['JAGX_BATCH_SIZE'] = '4'
os.environ['JAGX_GRAD_ACCUM'] = '8'
```

Then run:

```bash
python scripts/kaggle_train.py --skip-pip
```

Increase **one major variable at a time**. If you increase model size, sequence length, batch size and training steps simultaneously, it becomes difficult to identify what caused an out-of-memory error or unstable training.

## 7. Hugging Face access — what you actually need

For the current public OASST1 training path, you do **not** need to paste a Hugging Face token into the repository. The notebook already installs `datasets` and `huggingface_hub`, and the public dataset can be downloaded without exposing a private credential. fileciteturn3file0

You only need a Hugging Face token when a future training job needs authenticated access, such as private/gated resources or publishing artifacts to a Hugging Face repository.

If you need one:

1. Open your Hugging Face account settings.
2. Open **Access Tokens**.
3. Create a token with the smallest permission required.
4. Never put the token directly in Python code or commit it to GitHub.

Hugging Face recommends separate tokens for different uses and recommends fine-grained tokens where possible. Read-only access is appropriate when the job only needs to download private resources; write access is needed when the job must publish or modify repositories. citeturn0search0

## 8. If a Kaggle job needs the HF token

Use Kaggle's secret mechanism rather than hard-coding the token.

Conceptually, the training code should read:

```python
import os

HF_TOKEN = os.environ.get('HF_TOKEN')
```

Never do this:

```python
HF_TOKEN = 'hf_xxxxxxxxxxxxxxxxx'
```

The actual secret value should never appear in this README, a notebook cell, GitHub source code, or a commit.

## 9. GitHub secrets for CI/CD

If we later make GitHub Actions download private Hugging Face resources or publish checkpoints, create a repository secret instead of putting credentials in source code.

On GitHub:

**Repository → Settings → Secrets and variables → Actions → New repository secret**

For example:

```text
HF_TOKEN
```

Then a workflow can expose it to a job as an environment variable:

```yaml
env:
  HF_TOKEN: ${{ secrets.HF_TOKEN }}
```

GitHub encrypts repository secrets and only exposes a secret to a workflow when the workflow explicitly references it. GitHub also recommends minimum permissions for credentials. citeturn0search1turn0search4

Do not print the token in workflow logs.

## 10. What you should NOT put in GitHub

Never commit:

```text
hf_...
.env
private API keys
cloud credentials
Kaggle credentials
large checkpoints
raw private datasets
```

The repository already uses `.gitignore` rules for environment files and build artifacts. Keep credentials outside source control.

If a token is ever accidentally exposed, revoke/rotate it immediately. Hugging Face specifically recommends revoking leaked tokens rather than continuing to use them. citeturn0search0

## 11. Kaggle command cheat sheet

After the notebook has cloned the repository and installed dependencies, these are the main commands you will use.

### Verify the installation

```bash
pytest tests/ -q
jagx verify
```

### Inspect a model configuration

```bash
jagx inspect --config configs/kaggle.json
```

### Collect the gaming instruction seed corpus

```bash
python scripts/collect_gaming_corpus.py
```

### Run Kaggle training

```bash
python scripts/kaggle_train.py --skip-pip
```

### Train manually on a prepared dataset

```bash
jagx train --data data.jsonl --tokenizer path/to/tok --config configs/tiny.json --steps 200 --out-dir checkpoints/run1
```

### Generate from a trained checkpoint

```bash
jagx generate "Hello" --checkpoint checkpoints/run1/... --tokenizer path/to/tok
```

### Start the local API

```bash
jagx serve --checkpoint ... --tokenizer ...
```

The repository's quick-start commands are also available in the original project instructions. fileciteturn1file0

## 12. What happens after the first successful checkpoint

The next development sequence is:

1. **Verify checkpoint loading.**
2. **Run inference on real prompts.**
3. **Measure training/validation loss.**
4. **Evaluate instruction following.**
5. **Expand and clean the corpus.**
6. **Add the gaming specialist data where appropriate.**
7. **Increase model size.**
8. **Increase sequence length only when GPU memory allows it.**
9. **Add stronger evaluation suites.**
10. **Run repeated training experiments and keep the best validated checkpoint.**
11. **Only then move toward multi-GPU/distributed training.**

This is how we turn the repository from an architecture into a genuinely trained model instead of assuming that a large configuration file is already a large AI.

## 13. About the 200B+ goal

A configuration saying `200B` does not create a 200-billion-parameter model. The parameters must actually be allocated, initialized, trained, checkpointed, and evaluated.

The current Kaggle pipeline is therefore the **foundation**, not the final 200B infrastructure.

The long-term scale path is:

```text
Smoke test
   ↓
Kaggle small model
   ↓
Kaggle larger model
   ↓
Multi-GPU experiment
   ↓
Distributed training
   ↓
Large cluster training
   ↓
Large-scale checkpoint + evaluation
```

We should not skip the validation stages because a failed large run can waste enormous compute while hiding a data, tokenizer, numerical-stability, checkpointing, or batching problem.

## 14. Troubleshooting

### `CUDA available: False`

GPU is not enabled. Turn on Kaggle GPU acceleration and restart the session.

### `training batches are empty`

Do not bypass the error. Check the downloaded/prepared corpus and tokenizer preparation. The current launcher is designed to fail before the trainer if it cannot create a real batch. fileciteturn2file0

### CUDA out-of-memory

Reduce, in this order:

1. `JAGX_BATCH_SIZE`
2. model size/configuration
3. `JAGX_SEQ_LEN`

Then use gradient accumulation to recover some effective batch size.

### Kaggle session stops

Kaggle sessions are temporary. Treat Kaggle working storage as temporary too. Important checkpoints should be copied to a durable location after training rather than assuming the notebook session will remain available.

### GitHub is getting huge

Do not commit generated datasets or `.pt` checkpoints. Keep GitHub for source code, configurations, manifests and reproducible scripts.

## 15. Current recommended first run

For your next training attempt, do exactly this:

1. Open a new Kaggle Notebook.
2. Turn **Internet ON**.
3. Turn **GPU ON**.
4. Import `notebooks/jagx_kaggle_train.ipynb`.
5. Run every cell from top to bottom.
6. Leave the default `50,000` records and `1,000` steps for the first run.
7. Wait for the training/checkpoint cell to finish.
8. Run the final checkpoint inspection cell.
9. Save/copy the resulting checkpoint somewhere durable.
10. Come back with the training output/log if anything fails.

**Do not change the model to 200B yet. First prove that this complete pipeline produces a valid checkpoint.**

## Quick start

```bash
pip install -e ".[dev]"
pytest tests/ -q
jagx verify
jagx agent "summarize research notes"
jagx tool echo --args '{"message":"ok"}'
```

Train then generate:

```bash
jagx train --data data.jsonl --tokenizer path/to/tok --config configs/tiny.json --steps 200 --out-dir checkpoints/run1
jagx generate "Hello" --checkpoint checkpoints/run1/... --tokenizer path/to/tok
jagx serve --checkpoint ... --tokenizer ...
```

Inspect a size and train on Kaggle:

```bash
jagx inspect --config configs/kaggle.json
python scripts/collect_gaming_corpus.py
python scripts/kaggle_train.py --resume
```

See `docs/TRAIN_WELL.md`, `docs/KAGGLE_TRAINING.md`, `docs/USE_NOW.md`, `docs/FREE_TRAINING.md`, `docs/CAPABILITY_MATRIX.md`.
