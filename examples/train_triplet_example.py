"""Example script for training triplet loss occupations classifier."""

import os
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Check if environment is properly set up
try:
    import torch
    import transformers
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ Transformers version: {transformers.__version__}")
except ImportError as e:
    print("❌ Environment not properly set up!")
    print("Please activate the conda environment:")
    print("  conda activate nlp_project_dev")
    print(f"\nError: {e}")
    sys.exit(1)

from nlp_project.training.train_triplet_occupations import main as train_main


def run_triplet_training_example():
    """Run triplet loss training with advanced embedding learning."""
    print("=" * 80)
    print("🔺 TRIPLET LOSS OCCUPATIONS CLASSIFIER TRAINING")
    print("=" * 80)
    
    # Check if data exists
    data_dir = "/Users/tod/data/occupations"
    train_file = os.path.join(data_dir, "train_df.csv")
    val_file = os.path.join(data_dir, "val_df.csv")
    
    if not os.path.exists(train_file):
        print(f"❌ Training data not found at: {train_file}")
        return
    
    if not os.path.exists(val_file):
        print(f"❌ Validation data not found at: {val_file}")
        return
    
    print(f"✅ Found training data: {train_file}")
    print(f"✅ Found validation data: {val_file}")
    
    print("\n🔺 TRIPLET LOSS ADVANTAGES:")
    print("  🎯 Learns semantic similarity between occupations")
    print("  📏 Creates better embeddings for similar jobs")
    print("  🔍 Especially effective for few-shot learning")
    print("  💪 Helps with imbalanced classes by learning representations")
    
    print("\n🔧 TRIPLET TRAINING PARAMETERS:")
    print("  🔺 Loss function: Combined Triplet + Classification")
    print("  📊 Triplet margin: 0.5")
    print("  ⚖️  Loss weights: Triplet=1.0, Classification=1.0")
    print("  📐 Embedding dimension: 256")
    print("  📦 Batch size: 32 (larger for better triplet mining)")
    print("  📏 Max sequence length: 256")
    print("  📉 Learning rate: 1e-5")
    print("  🔄 Epochs: 15 (more epochs for embedding learning)")
    print("  🎯 Mining strategy: Hard negative mining after epoch 2")
    print("  ⏹️  Early stopping: 4 epochs patience")
    print("  🎯 Output: ./models/occupations_triplet_classifier")
    
    print("\n🧠 HOW TRIPLET LOSS WORKS:")
    print("  1. 📌 Anchor: Reference occupation (e.g., 'Software Engineer')")
    print("  2. ✅ Positive: Similar occupation (e.g., 'Data Scientist')")
    print("  3. ❌ Negative: Different occupation (e.g., 'Chef')")
    print("  4. 🎯 Goal: distance(anchor, positive) + margin < distance(anchor, negative)")
    print("  5. 🔍 Hard mining: Find hardest positives/negatives for better learning")
    
    print("\n" + "=" * 80)
    print("Starting triplet loss training...")
    print("=" * 80)
    
    # Override sys.argv to simulate command line arguments
    original_argv = sys.argv.copy()
    sys.argv = [
        "train_triplet_occupations.py",
        "--data_dir", data_dir,
        "--model_name", "bert-base-uncased",
        "--batch_size", "32",
        "--max_length", "256",
        "--learning_rate", "1e-5",
        "--epochs", "15",
        "--embedding_dim", "256",
        "--triplet_margin", "0.5",
        "--triplet_weight", "1.0",
        "--classification_weight", "1.0",
        "--output_dir", "./models/occupations_triplet_classifier",
        "--device", "auto",
        "--dropout", "0.3"
    ]
    
    try:
        train_main()
        print("\n" + "=" * 80)
        print("✅ Triplet loss training completed successfully!")
        print("📁 Model saved to: ./models/occupations_triplet_classifier")
        print("=" * 80)
        
        print("\n🔍 NEXT STEPS:")
        print("1. 📊 Evaluate the triplet model:")
        print("   python -m nlp_project.evaluation.evaluate \\")
        print("       --model_path ./models/occupations_triplet_classifier \\")
        print("       --test_data /Users/tod/data/occupations/test_df.csv")
        print()
        print("2. 📈 Compare with baseline and improved models:")
        print("   - Baseline F1: ~0.38")
        print("   - Improved F1: ~0.5+ (expected)")
        print("   - Triplet F1: ~0.6+ (expected with better embeddings)")
        print()
        print("3. 🔍 Analyze embeddings:")
        print("   - The model learns 256-dimensional embeddings")
        print("   - Similar occupations should cluster together")
        print("   - Dissimilar occupations should be far apart")
        
    except Exception as e:
        print(f"\n❌ Triplet training failed with error: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
    finally:
        # Restore original argv
        sys.argv = original_argv


if __name__ == "__main__":
    run_triplet_training_example()