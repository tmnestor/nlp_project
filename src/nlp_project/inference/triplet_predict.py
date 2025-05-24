"""Inference script for triplet loss-based occupations classification."""

import argparse
import os
from pathlib import Path
from typing import Any

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

from ..models.triplet_classifier import TripletClassifier


class TripletOccupationClassifier:
    """Wrapper class for triplet-based occupation classification inference."""

    def __init__(self, model_path: str, device: str = "auto"):
        """Initialize the triplet classifier.

        Args:
            model_path: Path to directory containing model files
            device: Device to use for inference
        """
        self.model_path = Path(model_path)

        # Set device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Load model
        self._load_model()

        # Load label mappings
        self._load_label_mappings()

    def _load_model(self):
        """Load the trained triplet model."""
        model_file = self.model_path / "best_model.pt"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        checkpoint = torch.load(model_file, map_location=self.device)
        model_config = checkpoint["model_config"]

        # Initialize model
        self.model = TripletClassifier(**model_config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"])

        # Store model config for reference
        self.model_config = model_config
        self.training_config = checkpoint.get("training_config", {})

        print(f"Triplet model loaded with accuracy: {checkpoint['accuracy']:.4f}")
        print(f"Embedding dimension: {model_config['embedding_dim']}")

    def _load_label_mappings(self):
        """Load label mappings."""
        mappings_file = self.model_path / "label_mappings.pt"
        if not mappings_file.exists():
            raise FileNotFoundError(f"Label mappings file not found: {mappings_file}")

        mappings = torch.load(mappings_file, map_location="cpu")
        self.label_to_guide = mappings["label_to_guide"]
        self.label_encoder = mappings["label_encoder"]

    def predict(
        self, text: str, return_embeddings: bool = False, return_probs: bool = False
    ) -> dict[str, Any]:
        """Predict occupation category for given text.

        Args:
            text: Input occupation title/description
            return_embeddings: Whether to return learned embeddings
            return_probs: Whether to return class probabilities

        Returns:
            Dictionary with prediction results
        """
        # Tokenize input
        inputs = self.tokenizer(
            text, truncation=True, padding="max_length", max_length=256, return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Make prediction
        with torch.no_grad():
            logits, embeddings = self.model(
                input_ids=input_ids, attention_mask=attention_mask, return_embeddings=True
            )
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

        # Get guide name
        guide_name = self.label_to_guide.get(predicted_class, "Unknown")

        result = {
            "input_text": text,
            "predicted_label": predicted_class,
            "predicted_guide": guide_name,
            "confidence": probabilities[0][predicted_class].item(),
        }

        if return_embeddings:
            result["embeddings"] = embeddings[0].cpu().numpy()

        if return_probs:
            # Get top 5 predictions
            top_probs, top_indices = torch.topk(probabilities[0], k=5)
            top_predictions = []
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices, strict=False)):
                guide = self.label_to_guide.get(idx.item(), "Unknown")
                top_predictions.append(
                    {"rank": i + 1, "label": idx.item(), "guide": guide, "probability": prob.item()}
                )
            result["top_predictions"] = top_predictions

        return result

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        """Get embeddings for multiple texts.

        Args:
            texts: List of occupation titles/descriptions

        Returns:
            Array of embeddings [num_texts, embedding_dim]
        """
        embeddings_list = []

        for text in texts:
            result = self.predict(text, return_embeddings=True)
            embeddings_list.append(result["embeddings"])

        return np.array(embeddings_list)

    def find_similar_occupations(
        self, query_text: str, reference_texts: list[str], top_k: int = 5
    ) -> list[dict]:
        """Find most similar occupations using learned embeddings.

        Args:
            query_text: Query occupation title
            reference_texts: List of reference occupation titles
            top_k: Number of most similar occupations to return

        Returns:
            List of similar occupations with similarity scores
        """
        # Get embeddings
        query_embedding = self.get_embeddings([query_text])
        reference_embeddings = self.get_embeddings(reference_texts)

        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, reference_embeddings)[0]

        # Get top-k most similar
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for i, idx in enumerate(top_indices):
            results.append(
                {
                    "rank": i + 1,
                    "text": reference_texts[idx],
                    "similarity": similarities[idx],
                    "prediction": self.predict(reference_texts[idx]),
                }
            )

        return results

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Predict for multiple texts.

        Args:
            texts: List of occupation titles/descriptions

        Returns:
            List of prediction dictionaries
        """
        return [self.predict(text) for text in texts]


def main():
    parser = argparse.ArgumentParser(description="Predict occupation categories with triplet model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/occupations_triplet_classifier",
        help="Path to trained triplet model directory",
    )
    parser.add_argument("--text", type=str, help="Text to classify")
    parser.add_argument("--file", type=str, help="File with texts to classify (one per line)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument(
        "--return_probs", action="store_true", help="Return top 5 predictions with probabilities"
    )
    parser.add_argument("--return_embeddings", action="store_true", help="Return learned embeddings")
    parser.add_argument("--find_similar", type=str, help="Find similar occupations to this query")
    parser.add_argument(
        "--reference_file", type=str, help="File with reference texts for similarity search"
    )

    args = parser.parse_args()

    if not args.text and not args.file and not args.find_similar:
        parser.error("Either --text, --file, or --find_similar must be provided")

    # Initialize classifier
    classifier = TripletOccupationClassifier(args.model_path, args.device)

    if args.text:
        # Single prediction
        result = classifier.predict(
            args.text, return_probs=args.return_probs, return_embeddings=args.return_embeddings
        )
        print(f"\n🎯 Input: {result['input_text']}")
        print(f"📊 Predicted Guide: {result['predicted_guide']}")
        print(f"🔍 Confidence: {result['confidence']:.4f}")

        if args.return_embeddings:
            print(f"🧠 Embedding shape: {result['embeddings'].shape}")
            print(f"📏 Embedding norm: {np.linalg.norm(result['embeddings']):.4f}")

        if args.return_probs:
            print("\n📈 Top 5 Predictions:")
            for pred in result["top_predictions"]:
                print(f"  {pred['rank']}. {pred['guide']} ({pred['probability']:.4f})")

    elif args.file:
        # Batch prediction
        with open(args.file) as f:
            texts = [line.strip() for line in f if line.strip()]

        results = classifier.predict_batch(texts)

        print(f"\n📊 Processed {len(results)} texts:")
        print("-" * 80)
        for result in results:
            print(f"Input: {result['input_text']}")
            print(f"Predicted: {result['predicted_guide']} (confidence: {result['confidence']:.4f})")
            print("-" * 80)

    elif args.find_similar:
        # Similarity search
        if not args.reference_file:
            parser.error("--reference_file is required for similarity search")

        with open(args.reference_file) as f:
            reference_texts = [line.strip() for line in f if line.strip()]

        similar_occupations = classifier.find_similar_occupations(
            args.find_similar, reference_texts, top_k=10
        )

        print(f"\n🔍 Most similar occupations to '{args.find_similar}':")
        print("-" * 80)
        for item in similar_occupations:
            pred = item["prediction"]
            print(f"{item['rank']}. {item['text']}")
            print(f"   Similarity: {item['similarity']:.4f}")
            print(f"   Category: {pred['predicted_guide']}")
            print(f"   Confidence: {pred['confidence']:.4f}")
            print()


if __name__ == "__main__":
    main()
