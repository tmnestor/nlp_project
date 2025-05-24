"""Dataset classes for NLP tasks."""

from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class TextDataset(Dataset):
    """Base dataset class for text data.

    Args:
        texts: List of input texts
        labels: Optional list of labels
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length

    Returns:
        Dictionary with input_ids, attention_mask, and optional labels
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer: PreTrainedTokenizer,
        labels: list[Any] | None = None,
        max_length: int = 512,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item
