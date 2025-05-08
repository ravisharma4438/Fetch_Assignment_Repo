# Multi-Task Learning with BERT

This repository shows a **minimal yet well-structured** example of using BERT for two tasks at once:

1. **Domain classification** (history / geography / health / technology)
2. **Sentiment analysis** (negative / neutral / positive)

It also exposes a tiny **sentence-embedding utility** so you can obtain the CLS-token embeddings for any sentence.

---

## Repository layout

```
.
├── app                     # Python package with all source code
│   ├── __init__.py
│   ├── data
│   │   ├── __init__.py
│   │   └── dataset.py      # Helpers for loading csv → Torch datasets
│   └── models
│       ├── __init__.py
│       ├── embedder.py     # Thin wrapper around HuggingFace BERT
│       └── heads.py        # Generic classification head(s)
│
├── data                    # Tiny example datasets used in the demo
│   ├── domain.csv
│   └── sentiment.csv
│
├── embed.py                # Example: produce embeddings for two sentences
├── train.py                # Multi-task training script
├── requirements.txt        # Python dependencies
├── Dockerfile              # Optional container recipe
├── write_up
│   └── write_up.docx       # Detailed design write-up & assignment answers
└── README.md               # You are here 🙂
```

---

## Quick start (local)

1. **Create and activate a virtual environment** (recommended).

```bash
python -m venv .venv
source .venv/bin/activate
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run the sentence embedder**:

```bash
python embed.py
```

You should see output in the following format (vector shortened for readability):

```
sentence: There is a stack of papers on the table.
embedding: [0.03, -0.12, ..., 0.45]

sentence: The largest mountain in the world is Mount Everest.
embedding: [-0.22, 0.07, ..., 0.11]
```

4. **Train the multi-task model** (uses the tiny demo datasets under `data/`):

```bash
python train.py --num_epochs 3 --output_dir outputs
```

---

## Using the Docker image (optional)

Build the image:

```bash
docker build -t bert-mtl .
```

Run an embedding example (default):

```bash
# prints embeddings
docker run --rm bert-mtl
```

Run training instead (override the default command):

```bash
# trains for 3 epochs and stores outputs inside the container
docker run --rm bert-mtl train.py --num_epochs 3

# to persist checkpoints to host machine:
docker run --rm -v $(pwd)/outputs:/app/outputs bert-mtl train.py --num_epochs 3 --output_dir outputs
```

---

## Notes

* The datasets provided are **toy examples** – they are only meant to demonstrate code execution. Replace them with real data for meaningful results.
* The HuggingFace model weights are downloaded at first run and cached under `~/.cache/huggingface/`.
* The training script now stores the **fine-tuned BERT encoder _and_ the task-specific heads** in one checkpoint file for easy reuse.
* A detailed write-up of design choices and assignment Q&A lives at `write_up/write_up.docx`. 