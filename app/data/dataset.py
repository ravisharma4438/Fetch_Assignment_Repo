import pandas as pd
import torch
from torch.utils.data import TensorDataset
from transformers import PreTrainedTokenizer
from typing import Dict


def load_task_csv(
    path: str,
    text_col: str,
    label_col: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    label2id: Dict[str, int],
) -> TensorDataset:
    """Read a CSV file and convert it into a `TensorDataset` ready for PyTorch.

    Args:
        path: Path to the CSV file.
        text_col: Name of the column containing sentences.
        label_col: Name of the column containing labels (str).
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length for padding/truncation.
        label2id: Mapping string → numeric id.

    Returns:
        `TensorDataset` with `(input_ids, attention_mask, label)`.
    """
    df = pd.read_csv(path)
    sentences = df[text_col].tolist()
    labels_str = df[label_col].tolist()
    numeric_labels = [label2id[l] for l in labels_str]

    encodings = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    dataset = TensorDataset(
        encodings.input_ids,
        encodings.attention_mask,
        torch.tensor(numeric_labels, dtype=torch.long),
    )
    return dataset 