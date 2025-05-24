"""Training script for triplet loss-based occupations classification."""

import argparse
import os
from collections import Counter
from pathlib import Path

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from ..data.occupations_loader import (
    get_label_mappings,
    load_occupations_data,
)
from ..models.triplet_classifier import CombinedLoss, TripletClassifier


def create_balanced_dataloader(
    data_path: str,
    tokenizer_name: str = "bert-base-uncased",
    batch_size: int = 16,
    max_length: int = 256,
    shuffle: bool = True,
    use_class_weights: bool = True,
) -> tuple:
    """Create balanced DataLoader using weighted sampling."""
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from ..data.dataset import TextDataset

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

    # Create weighted sampler for class balance
    sampler = None
    if use_class_weights and shuffle:
        class_counts = Counter(labels)
        # Calculate weights for each sample (inverse frequency)
        weights = [1.0 / class_counts[label] for label in labels]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False  # Don't shuffle when using sampler

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
    )

    return dataloader, num_classes


def get_class_weights(labels: list) -> torch.Tensor:
    """Compute class weights for loss function."""
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight("balanced", classes=unique_labels, y=labels)
    return torch.FloatTensor(class_weights)


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    scheduler,
    device: str,
    combined_loss: CombinedLoss,
    epoch: int,
) -> dict:
    """Train for one epoch with combined triplet + classification loss."""
    model.train()
    total_losses = {
        "total_loss": 0.0,
        "classification_loss": 0.0,
        "triplet_loss": 0.0,
        "num_triplets": 0,
    }
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass - get both logits and embeddings
        logits, embeddings = model(
            input_ids=input_ids, attention_mask=attention_mask, return_embeddings=True
        )

        # Compute combined loss with progressive triplet introduction
        # Epoch 1: Classification only
        # Epoch 2+: Add triplet loss gradually
        if epoch == 1:
            # Classification only for first epoch
            cls_loss = combined_loss.classification_loss(logits, labels)
            loss = cls_loss
            loss_dict = {
                "total_loss": loss.item(),
                "classification_loss": cls_loss.item(),
                "triplet_loss": 0.0,
                "num_triplets": 0,
            }
        else:
            # Combined loss with triplet
            loss, loss_dict = combined_loss(
                logits=logits,
                embeddings=embeddings,
                labels=labels,
                use_hard_mining=(epoch >= 3),  # Start hard mining after 3 epochs
            )

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Accumulate losses
        for key, value in loss_dict.items():
            total_losses[key] += value
        num_batches += 1

        # Debug: Print first batch info for epoch 1
        if epoch == 1 and num_batches == 1:
            predictions = torch.argmax(logits, dim=1)
            print("Debug - Training first batch:")
            print(f"  Predictions: {predictions[:10].cpu().numpy()}")
            print(f"  Labels: {labels[:10].cpu().numpy()}")
            print(f"  Loss: {loss_dict['total_loss']:.4f}")
            print(
                f"  Logits stats: min={logits.min().item():.3f}, "
                f"max={logits.max().item():.3f}, std={logits.std().item():.3f}"
            )
            print(f"  Logits for class 35: {logits[0, 35].item():.3f}")
            print(f"  Unique predictions: {torch.unique(predictions).cpu().numpy()}")

        # Update progress bar
        progress_bar.set_postfix(
            {
                "Loss": f"{loss_dict['total_loss']:.3f}",
                "Cls": f"{loss_dict['classification_loss']:.3f}",
                "Trip": f"{loss_dict['triplet_loss']:.3f}",
                "Trips": loss_dict["num_triplets"],
            }
        )

    # Average losses
    for key in total_losses:
        if key != "num_triplets":
            total_losses[key] /= num_batches
        else:
            total_losses[key] = int(total_losses[key] / num_batches)

    return total_losses


def evaluate(
    model: nn.Module,
    dataloader,
    device: str,
    combined_loss: CombinedLoss,
    epoch: int = 1,
) -> tuple[dict, float]:
    """Evaluate model with combined loss."""
    model.eval()
    total_losses = {
        "total_loss": 0.0,
        "classification_loss": 0.0,
        "triplet_loss": 0.0,
        "num_triplets": 0,
    }
    correct = 0
    total = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            logits, embeddings = model(
                input_ids=input_ids, attention_mask=attention_mask, return_embeddings=True
            )

            # Compute loss (match training logic)
            if epoch == 1:
                # Classification only for first epoch
                cls_loss = combined_loss.classification_loss(logits, labels)
                loss = cls_loss
                loss_dict = {
                    "total_loss": loss.item(),
                    "classification_loss": cls_loss.item(),
                    "triplet_loss": 0.0,
                    "num_triplets": 0,
                }
            else:
                # Combined loss for later epochs
                loss, loss_dict = combined_loss(
                    logits=logits, embeddings=embeddings, labels=labels, use_hard_mining=False
                )

            # Accumulate losses
            for key, value in loss_dict.items():
                total_losses[key] += value
            num_batches += 1

            # Compute accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Debug: Print first batch predictions vs labels
            if num_batches == 1:
                print("Debug - Validation first batch:")
                print(f"  Predictions: {predictions[:10].cpu().numpy()}")
                print(f"  Labels: {labels[:10].cpu().numpy()}")
                print(f"  Pred range: [{predictions.min().item()}, {predictions.max().item()}]")
                print(f"  Label range: [{labels.min().item()}, {labels.max().item()}]")
                print(f"  Logits shape: {logits.shape}")
                print(f"  Logits stats: min={logits.min().item():.3f}, max={logits.max().item():.3f}")
                print(f"  Unique predictions: {torch.unique(predictions).cpu().numpy()}")
                print(f"  Num correct in batch: {(predictions == labels).sum().item()}/{labels.size(0)}")

    # Average losses
    for key in total_losses:
        if key != "num_triplets":
            total_losses[key] /= num_batches
        else:
            total_losses[key] = int(total_losses[key] / num_batches)

    accuracy = correct / total
    return total_losses, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train triplet loss occupations classifier")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Users/tod/data/occupations",
        help="Directory containing train_df.csv, val_df.csv, test_df.csv",
    )
    parser.add_argument(
        "--model_name", type=str, default="bert-base-uncased", help="HuggingFace model name"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size (increased for triplet mining)"
    )
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--triplet_margin", type=float, default=0.5, help="Triplet loss margin")
    parser.add_argument("--triplet_weight", type=float, default=0.1, help="Weight for triplet loss")
    parser.add_argument(
        "--classification_weight", type=float, default=1.0, help="Weight for classification loss"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/occupations_triplet_classifier",
        help="Output directory for saved model",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Create data loaders (disable weighted sampling to prevent mode collapse)
    train_loader, num_classes = create_balanced_dataloader(
        data_path=os.path.join(args.data_dir, "train_df.csv"),
        tokenizer_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
        use_class_weights=False,  # Disable weighted sampling
    )

    val_loader, _ = create_balanced_dataloader(
        data_path=os.path.join(args.data_dir, "val_df.csv"),
        tokenizer_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False,
        use_class_weights=False,
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Number of classes: {num_classes}")

    # Debug: Check first batch of training data
    print("\nDebugging data loading...")
    first_batch = next(iter(train_loader))
    print(f"First batch labels: {first_batch['labels'][:10].numpy()}")
    print(f"Labels range: [{first_batch['labels'].min().item()}, {first_batch['labels'].max().item()}]")
    print(f"Batch size: {first_batch['labels'].shape[0]}")

    # Check validation data too
    first_val_batch = next(iter(val_loader))
    print(f"First val batch labels: {first_val_batch['labels'][:10].numpy()}")
    print(
        f"Val labels range: "
        f"[{first_val_batch['labels'].min().item()}, {first_val_batch['labels'].max().item()}]"
    )

    # Analyze class distribution
    train_texts, train_labels, _ = load_occupations_data(os.path.join(args.data_dir, "train_df.csv"))

    # Debug class distribution
    from collections import Counter

    class_counts = Counter(train_labels)
    print(f"Class distribution (first 10): {dict(list(class_counts.most_common(10)))}")
    print(f"Class distribution (last 10): {dict(list(class_counts.most_common()[-10:]))}")
    print(f"Class 35 count: {class_counts.get(35, 0)}")

    # NO CLASS WEIGHTS - use uniform weighting to prevent mode collapse
    class_weights = None
    print("Using NO class weights to prevent mode collapse")

    # Initialize triplet classifier
    model = TripletClassifier(
        model_name=args.model_name,
        num_classes=num_classes,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        freeze_bert=False,
    )
    model.to(device)

    # Initialize combined loss
    combined_loss = CombinedLoss(
        triplet_margin=args.triplet_margin,
        classification_weight=args.classification_weight,
        triplet_weight=args.triplet_weight,
        class_weights=class_weights,
    )

    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    print("\nTraining Configuration:")
    print(f"  Embedding dimension: {args.embedding_dim}")
    print(f"  Triplet margin: {args.triplet_margin}")
    print(f"  Triplet weight: {args.triplet_weight}")
    print(f"  Classification weight: {args.classification_weight}")
    print(f"  Batch size: {args.batch_size}")

    # Training loop
    best_val_accuracy = 0.0
    patience = 4
    patience_counter = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, scheduler, device, combined_loss, epoch + 1
        )
        print(
            f"Train - Total: {train_losses['total_loss']:.4f}, "
            f"Cls: {train_losses['classification_loss']:.4f}, "
            f"Trip: {train_losses['triplet_loss']:.4f}, "
            f"Triplets: {train_losses['num_triplets']}"
        )

        # Validate
        val_losses, val_accuracy = evaluate(model, val_loader, device, combined_loss, epoch + 1)
        print(
            f"Val - Total: {val_losses['total_loss']:.4f}, "
            f"Cls: {val_losses['classification_loss']:.4f}, "
            f"Trip: {val_losses['triplet_loss']:.4f}, "
            f"Accuracy: {val_accuracy:.4f}"
        )

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0

            # Create output directory
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save model
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "model_name": args.model_name,
                        "num_classes": num_classes,
                        "embedding_dim": args.embedding_dim,
                        "dropout": args.dropout,
                        "freeze_bert": False,
                    },
                    "training_config": {
                        "triplet_margin": args.triplet_margin,
                        "triplet_weight": args.triplet_weight,
                        "classification_weight": args.classification_weight,
                    },
                    "accuracy": val_accuracy,
                    "epoch": epoch,
                    "class_weights": class_weights,
                },
                output_path / "best_model.pt",
            )

            print(f"✅ Saved new best model with accuracy: {val_accuracy:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {patience} epochs without improvement")
            break

    # Save label mappings
    label_to_guide, label_encoder = get_label_mappings(os.path.join(args.data_dir, "train_df.csv"))

    torch.save(
        {
            "label_to_guide": label_to_guide,
            "label_encoder": label_encoder,
        },
        Path(args.output_dir) / "label_mappings.pt",
    )

    print(f"\nTraining completed! Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
