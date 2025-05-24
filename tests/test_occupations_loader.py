"""Tests for occupations data loader."""

import os
import tempfile

import pandas as pd
import pytest

from src.nlp_project.data.occupations_loader import (
    create_occupations_dataloader,
    get_label_mappings,
    load_occupations_data,
)


@pytest.fixture
def sample_data():
    """Create sample occupations data for testing."""
    data = {
        'text': [
            'Software Engineer',
            'Medical Doctor',
            'Teacher',
            'Lawyer',
            'Invalid Row'
        ],
        'label': [1, 2, 3, 4, 'invalid'],
        'guide': [
            'Information Technology Workers',
            'Doctor, Specialist Or Other Medical Professional',
            'Education Workers',
            'Lawyers',
            'Invalid Guide'
        ]
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_csv_file(sample_data):
    """Create temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


def test_load_occupations_data(temp_csv_file):
    """Test loading occupations data from CSV."""
    texts, labels, guide_names = load_occupations_data(temp_csv_file)

    # Should filter out invalid rows
    assert len(texts) == 4
    assert len(labels) == 4
    assert len(guide_names) == 4

    # Check data types
    assert all(isinstance(text, str) for text in texts)
    assert all(isinstance(label, int) for label in labels)
    assert all(isinstance(guide, str) for guide in guide_names)

    # Check specific values
    assert 'Software Engineer' in texts
    assert 1 in labels
    assert 'Information Technology Workers' in guide_names


def test_create_occupations_dataloader(temp_csv_file):
    """Test creating DataLoader for occupations data."""
    dataloader, num_classes = create_occupations_dataloader(
        data_path=temp_csv_file,
        batch_size=2,
        max_length=64,
        shuffle=False
    )

    # Check DataLoader properties
    assert dataloader.batch_size == 2
    assert len(dataloader.dataset) == 4  # Valid rows only
    assert num_classes == 4  # Unique labels: 1, 2, 3, 4

    # Check first batch
    batch = next(iter(dataloader))
    assert 'input_ids' in batch
    assert 'attention_mask' in batch
    assert 'labels' in batch

    # Check tensor shapes
    assert batch['input_ids'].shape[1] == 64  # max_length
    assert batch['attention_mask'].shape[1] == 64
    assert len(batch['labels']) <= 2  # batch_size


def test_get_label_mappings(temp_csv_file):
    """Test getting label mappings."""
    label_to_guide, label_encoder = get_label_mappings(temp_csv_file)

    # Check label to guide mapping
    assert isinstance(label_to_guide, dict)
    assert len(label_to_guide) == 4
    assert label_to_guide[1] == 'Information Technology Workers'
    assert label_to_guide[2] == 'Doctor, Specialist Or Other Medical Professional'

    # Check label encoder
    assert hasattr(label_encoder, 'classes_')
    assert len(label_encoder.classes_) == 4


def test_empty_dataloader():
    """Test behavior with empty or invalid data."""
    # Create empty CSV
    empty_data = pd.DataFrame({'text': [], 'label': [], 'guide': []})

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        empty_data.to_csv(f.name, index=False)
        temp_file = f.name

    try:
        texts, labels, guide_names = load_occupations_data(temp_file)
        assert len(texts) == 0
        assert len(labels) == 0
        assert len(guide_names) == 0
    finally:
        os.unlink(temp_file)


def test_dataloader_batch_consistency(temp_csv_file):
    """Test that DataLoader produces consistent batches."""
    dataloader, _ = create_occupations_dataloader(
        data_path=temp_csv_file,
        batch_size=2,
        shuffle=False
    )

    # Get all batches
    all_batches = list(dataloader)

    # Check that all samples are covered
    total_samples = sum(len(batch['labels']) for batch in all_batches)
    assert total_samples == 4  # Valid rows

    # Check tensor consistency
    for batch in all_batches:
        batch_size = len(batch['labels'])
        assert batch['input_ids'].shape[0] == batch_size
        assert batch['attention_mask'].shape[0] == batch_size
