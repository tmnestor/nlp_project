"""Example script for training improved occupations classifier."""

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

from nlp_project.training.train_occupations_improved import main as train_main


def run_improved_training_example():
    """Run improved training with class balancing and better hyperparameters."""
    print("=" * 70)
    print("IMPROVED OCCUPATIONS CLASSIFIER TRAINING")
    print("=" * 70)
    
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
    
    print("\n🔧 IMPROVED TRAINING PARAMETERS:")
    print("  📊 Class imbalance handling: Weighted sampling + class weights")
    print("  📏 Max sequence length: 256 (increased for longer job titles)")
    print("  📉 Learning rate: 1e-5 (reduced for stable training)")
    print("  🔄 Epochs: 10 (increased with early stopping)")
    print("  💧 Dropout: 0.3 (increased to prevent overfitting)")
    print("  ⚖️  Weight decay: 0.01 (L2 regularization)")
    print("  ⏹️  Early stopping: 3 epochs patience")
    print("  🎯 Output: ./models/occupations_classifier_improved")
    
    print("\n" + "=" * 70)
    print("Starting improved training...")
    print("=" * 70)
    
    # Override sys.argv to simulate command line arguments
    original_argv = sys.argv.copy()
    sys.argv = [
        "train_occupations_improved.py",
        "--data_dir", data_dir,
        "--model_name", "bert-base-uncased",
        "--batch_size", "16",
        "--max_length", "256",
        "--learning_rate", "1e-5",
        "--epochs", "10",
        "--output_dir", "./models/occupations_classifier_improved",
        "--device", "auto",
        "--use_class_weights",
        "--dropout", "0.3"
    ]
    
    try:
        train_main()
        print("\n" + "=" * 70)
        print("✅ Improved training completed successfully!")
        print("📁 Model saved to: ./models/occupations_classifier_improved")
        print("=" * 70)
        
        print("\n🔍 NEXT STEPS:")
        print("1. Run evaluation:")
        print("   python -m nlp_project.evaluation.evaluate \\")
        print("       --model_path ./models/occupations_classifier_improved \\")
        print("       --test_data /Users/tod/data/occupations/test_df.csv")
        print("\n2. Compare with baseline results to see improvement")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
    finally:
        # Restore original argv
        sys.argv = original_argv


if __name__ == "__main__":
    run_improved_training_example()