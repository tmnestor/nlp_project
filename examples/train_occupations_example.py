"""Example script for training occupations classifier."""

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
    print("\nOr create/update the environment:")
    print("  mamba env create -f environment-dev.yml")
    print("  conda activate nlp_project_dev")
    print(f"\nError: {e}")
    sys.exit(1)

from nlp_project.training.train_occupations import main as train_main


def run_training_example():
    """Run training with example parameters."""
    print("=" * 60)
    print("OCCUPATIONS CLASSIFIER TRAINING EXAMPLE")
    print("=" * 60)
    
    # Set up arguments for training
    import argparse
    
    # Create a mock args object with example parameters
    class Args:
        def __init__(self):
            self.data_dir = "/Users/tod/data/occupations"
            self.model_name = "bert-base-uncased"
            self.batch_size = 16
            self.max_length = 128
            self.learning_rate = 2e-5
            self.epochs = 3
            self.output_dir = "./models/occupations_classifier"
            self.device = "auto"
    
    # Check if data exists
    data_dir = "/Users/tod/data/occupations"
    train_file = os.path.join(data_dir, "train_df.csv")
    val_file = os.path.join(data_dir, "val_df.csv")
    
    if not os.path.exists(train_file):
        print(f"❌ Training data not found at: {train_file}")
        print("Please ensure the occupations dataset is available.")
        return
    
    if not os.path.exists(val_file):
        print(f"❌ Validation data not found at: {val_file}")
        print("Please ensure the occupations dataset is available.")
        return
    
    print(f"✅ Found training data: {train_file}")
    print(f"✅ Found validation data: {val_file}")
    
    print("\nTraining Parameters:")
    print(f"  Data Directory: {data_dir}")
    print(f"  Model: bert-base-uncased")
    print(f"  Batch Size: 16")
    print(f"  Max Length: 128")
    print(f"  Learning Rate: 2e-5")
    print(f"  Epochs: 3")
    print(f"  Output Directory: ./models/occupations_classifier")
    
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    # Override sys.argv to simulate command line arguments
    original_argv = sys.argv.copy()
    sys.argv = [
        "train_occupations.py",
        "--data_dir", data_dir,
        "--model_name", "bert-base-uncased",
        "--batch_size", "16",
        "--max_length", "128",
        "--learning_rate", "2e-5",
        "--epochs", "3",
        "--output_dir", "./models/occupations_classifier",
        "--device", "auto"
    ]
    
    try:
        train_main()
        print("\n" + "=" * 60)
        print("✅ Training completed successfully!")
        print("Model saved to: ./models/occupations_classifier")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("=" * 60)
        import traceback
        print("Full traceback:")
        traceback.print_exc()
    finally:
        # Restore original argv
        sys.argv = original_argv


if __name__ == "__main__":
    run_training_example()