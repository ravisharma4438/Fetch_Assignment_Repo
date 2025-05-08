"""A thin wrapper around HuggingFace BERT that returns CLS embeddings."""

from typing import List, Union

import torch
from transformers import AutoModel, AutoTokenizer


class BertEmbedder:
    """Encapsulates tokenizer + encoder and exposes an `.encode` method."""

    def __init__(self, model_name: str = "bert-base-uncased", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        # We will not train BERT parameters via `.encode`, set eval mode
        self.model.eval()

    @torch.inference_mode()
    def encode(self, sentences: Union[str, List[str]]) -> torch.Tensor:
        """Return CLS-token embeddings for the given sentence(s).

        Args:
            sentences: A string or list of strings.
        Returns:
            A tensor of shape `(batch, hidden_size)` on **CPU**.
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        enc = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = enc.input_ids.to(self.device)
        attention_mask = enc.attention_mask.to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu()  # CLS token
        return embeddings 