"""Evaluation script for occupations classification."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from ..data.occupations_loader import create_occupations_dataloader
from ..models.classifier import TextClassifier


def evaluate_model(model_path: str, test_data_path: str, device: str = "auto") -> dict:
    """Evaluate trained model on test data.

    Args:
        model_path: Path to trained model directory
        test_data_path: Path to test CSV file
        device: Device to use for evaluation

    Returns:
        Dictionary with evaluation metrics
    """
    model_path = Path(model_path)

    # Set device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Load model
    model_file = model_path / "best_model.pt"
    checkpoint = torch.load(model_file, map_location=device)
    model_config = checkpoint["model_config"]

    model = TextClassifier(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load test data
    test_loader, num_classes = create_occupations_dataloader(
        data_path=test_data_path,
        tokenizer_name=model_config["model_name"],
        batch_size=32,
        max_length=128,
        shuffle=False,
    )

    # Load label mappings
    mappings_file = model_path / "label_mappings.pt"
    mappings = torch.load(mappings_file, map_location="cpu")
    label_to_guide = mappings["label_to_guide"]

    # Collect predictions
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)

    # Get unique labels present in test data
    unique_labels = sorted(set(all_labels))
    target_names = [label_to_guide.get(label, f"Label_{label}") for label in unique_labels]

    # Classification report
    report = classification_report(
        all_labels,
        all_predictions,
        labels=unique_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions, labels=unique_labels)

    # Calculate confidence statistics
    max_probs = np.max(all_probabilities, axis=1)
    confidence_stats = {
        "mean_confidence": float(np.mean(max_probs)),
        "median_confidence": float(np.median(max_probs)),
        "std_confidence": float(np.std(max_probs)),
    }

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confidence_stats": confidence_stats,
        "label_mapping": label_to_guide,
        "unique_labels": unique_labels,
        "target_names": target_names,
        "num_test_samples": len(all_labels),
    }


def print_evaluation_results(results: dict):
    """Print evaluation results in a formatted way."""
    print("=" * 80)
    print("OCCUPATION CLASSIFICATION EVALUATION RESULTS")
    print("=" * 80)

    print("\nDataset Statistics:")
    print(f"  Number of test samples: {results['num_test_samples']}")
    print(f"  Number of unique classes: {len(results['unique_labels'])}")

    print("\nOverall Performance:")
    print(f"  Accuracy: {results['accuracy']:.4f}")

    print("\nConfidence Statistics:")
    print(f"  Mean confidence: {results['confidence_stats']['mean_confidence']:.4f}")
    print(f"  Median confidence: {results['confidence_stats']['median_confidence']:.4f}")
    print(f"  Std confidence: {results['confidence_stats']['std_confidence']:.4f}")

    print("\nDetailed Classification Report:")
    print("-" * 80)

    # Print per-class metrics
    report = results["classification_report"]
    for i, label in enumerate(results["unique_labels"]):
        class_name = results["target_names"][i]
        label_str = str(label)
        if label_str in report:
            metrics = report[label_str]
            print(
                f"{class_name[:50]:50} | Precision: {metrics['precision']:.3f} | "
                f"Recall: {metrics['recall']:.3f} | F1: {metrics['f1-score']:.3f} | "
                f"Support: {metrics['support']:3d}"
            )

    print("-" * 80)
    print(
        f"{'Macro Avg':50} | Precision: {report['macro avg']['precision']:.3f} | "
        f"Recall: {report['macro avg']['recall']:.3f} | F1: {report['macro avg']['f1-score']:.3f}"
    )
    print(
        f"{'Weighted Avg':50} | Precision: {report['weighted avg']['precision']:.3f} | "
        f"Recall: {report['weighted avg']['recall']:.3f} | F1: {report['weighted avg']['f1-score']:.3f}"
    )


def save_evaluation_results(results: dict, output_path: str):
    """Save evaluation results to files."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed metrics as JSON-like format
    metrics_file = output_path / "evaluation_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write("OCCUPATION CLASSIFICATION EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Number of test samples: {results['num_test_samples']}\n")
        f.write(f"Number of classes: {len(results['unique_labels'])}\n\n")

        f.write("Per-class metrics:\n")
        report = results["classification_report"]
        for i, label in enumerate(results["unique_labels"]):
            class_name = results["target_names"][i]
            label_str = str(label)
            if label_str in report:
                metrics = report[label_str]
                f.write(
                    f"{class_name}: Precision={metrics['precision']:.3f}, "
                    f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}, "
                    f"Support={metrics['support']}\n"
                )

    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(
        results["confusion_matrix"],
        index=[f"True_{name}" for name in results["target_names"]],
        columns=[f"Pred_{name}" for name in results["target_names"]],
    )
    cm_df.to_csv(output_path / "confusion_matrix.csv")

    print(f"Evaluation results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate occupations classifier")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/occupations_classifier",
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="/Users/tod/data/occupations/test_df.csv",
        help="Path to test data CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device to use")

    args = parser.parse_args()

    print(f"Evaluating model: {args.model_path}")
    print(f"Test data: {args.test_data}")
    print(f"Device: {args.device}")

    # Run evaluation
    results = evaluate_model(args.model_path, args.test_data, args.device)

    # Print results
    print_evaluation_results(results)

    # Save results
    save_evaluation_results(results, args.output_dir)


if __name__ == "__main__":
    main()
