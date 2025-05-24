"""Tests for text classifier model."""

import pytest
import torch

from src.nlp_project.models.classifier import TextClassifier


@pytest.fixture
def sample_model():
    """Create a sample TextClassifier for testing."""
    return TextClassifier(
        model_name="bert-base-uncased",
        num_classes=5,
        dropout=0.1,
        freeze_bert=False
    )


def test_model_initialization(sample_model):
    """Test model initialization."""
    assert sample_model.bert is not None
    assert sample_model.dropout is not None
    assert sample_model.classifier is not None
    
    # Check classifier output size
    assert sample_model.classifier.out_features == 5


def test_model_forward_pass(sample_model):
    """Test forward pass with sample input."""
    batch_size = 2
    seq_length = 128
    
    # Create sample inputs
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    # Forward pass
    outputs = sample_model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Check output shape
    assert outputs.shape == (batch_size, 5)
    assert outputs.dtype == torch.float32


def test_model_forward_without_attention_mask(sample_model):
    """Test forward pass without attention mask."""
    batch_size = 2
    seq_length = 128
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    
    # Forward pass without attention mask
    outputs = sample_model(input_ids=input_ids)
    
    # Should still work
    assert outputs.shape == (batch_size, 5)


def test_model_with_frozen_bert():
    """Test model with frozen BERT weights."""
    model = TextClassifier(
        model_name="bert-base-uncased",
        num_classes=3,
        freeze_bert=True
    )
    
    # Check that BERT parameters are frozen
    for param in model.bert.parameters():
        assert not param.requires_grad
    
    # Check that classifier parameters are not frozen
    for param in model.classifier.parameters():
        assert param.requires_grad


def test_model_different_num_classes():
    """Test model with different number of classes."""
    for num_classes in [2, 10, 50]:
        model = TextClassifier(
            model_name="bert-base-uncased",
            num_classes=num_classes
        )
        
        batch_size = 1
        seq_length = 64
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        
        outputs = model(input_ids=input_ids)
        assert outputs.shape == (batch_size, num_classes)


def test_model_training_mode():
    """Test model behavior in training vs eval mode."""
    model = TextClassifier(
        model_name="bert-base-uncased",
        num_classes=5
    )
    
    batch_size = 2
    seq_length = 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    
    # Training mode
    model.train()
    assert model.training
    outputs_train = model(input_ids=input_ids)
    
    # Eval mode
    model.eval()
    assert not model.training
    outputs_eval = model(input_ids=input_ids)
    
    # Both should produce outputs of same shape
    assert outputs_train.shape == outputs_eval.shape


def test_model_gradient_flow():
    """Test that gradients flow properly through the model."""
    model = TextClassifier(
        model_name="bert-base-uncased",
        num_classes=3,
        freeze_bert=False
    )
    
    batch_size = 1
    seq_length = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    target = torch.randint(0, 3, (batch_size,))
    
    # Forward pass
    outputs = model(input_ids=input_ids)
    loss = torch.nn.CrossEntropyLoss()(outputs, target)
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist
    assert model.classifier.weight.grad is not None
    assert model.classifier.bias.grad is not None