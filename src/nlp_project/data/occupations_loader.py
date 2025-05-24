"""Data loader for occupations classification dataset."""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .dataset import TextDataset


def load_occupations_data(data_path: str) -> tuple[list[str], list[int], list[str]]:
    """Load occupations dataset from CSV file.

    Args:
        data_path: Path to CSV file with columns: text, label, guide

    Returns:
        Tuple of (texts, labels, guide_names)
    """
    df = pd.read_csv(data_path)

    # Clean the data - remove rows with parsing issues
    df = df.dropna(subset=["text", "label"])
    df = df[df["label"].astype(str).str.isdigit()]

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    guide_names = df["guide"].astype(str).tolist()

    return texts, labels, guide_names


def create_occupations_dataloader(
    data_path: str,
    tokenizer_name: str = "bert-base-uncased",
    batch_size: int = 16,
    max_length: int = 128,
    shuffle: bool = True,
) -> tuple[DataLoader, int]:
    """Create DataLoader for occupations dataset.

    Args:
        data_path: Path to CSV file
        tokenizer_name: HuggingFace tokenizer name
        batch_size: Batch size for DataLoader
        max_length: Maximum sequence length
        shuffle: Whether to shuffle data

    Returns:
        Tuple of (DataLoader, num_classes)
    """
    # Load data
    texts, labels, _ = load_occupations_data(data_path)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Get number of unique classes
    num_classes = len(set(labels))

    # Create dataset
    dataset = TextDataset(
        texts=texts,
        tokenizer=tokenizer,
        labels=labels,
        max_length=max_length,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Avoid multiprocessing issues
    )

    return dataloader, num_classes


def get_label_mappings(data_path: str) -> tuple[dict[int, str], LabelEncoder]:
    """Get label mappings for the occupations dataset.

    Args:
        data_path: Path to CSV file

    Returns:
        Tuple of (label_to_guide dict, label_encoder)
    """
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["label", "guide"])
    df = df[df["label"].astype(str).str.isdigit()]

    # Create mapping from label to guide name
    label_to_guide = dict(zip(df["label"].astype(int), df["guide"].astype(str), strict=False))

    # Create label encoder for consistent mapping
    label_encoder = LabelEncoder()
    label_encoder.fit(df["label"].astype(int))

    return label_to_guide, label_encoder
