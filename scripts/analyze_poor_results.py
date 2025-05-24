"""Script to analyze why the model performance is poor."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def analyze_class_distribution():
    """Analyze class distribution across train/val/test splits."""
    print("=" * 80)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    data_dir = "/Users/tod/data/occupations"
    
    for split in ['train_df', 'val_df', 'test_df']:
        file_path = f"{data_dir}/{split}.csv"
        df = pd.read_csv(file_path)
        
        print(f"\n📊 {split.upper()} SPLIT:")
        print(f"   Total samples: {len(df)}")
        print(f"   Unique classes: {df['label'].nunique()}")
        
        class_counts = df['label'].value_counts().sort_index()
        print(f"   Classes with <5 samples: {(class_counts < 5).sum()}")
        print(f"   Classes with <10 samples: {(class_counts < 10).sum()}")
        print(f"   Min samples per class: {class_counts.min()}")
        print(f"   Max samples per class: {class_counts.max()}")
        print(f"   Imbalance ratio: {class_counts.max() / class_counts.min():.1f}:1")


def analyze_text_characteristics():
    """Analyze text characteristics that might affect performance."""
    print("\n" + "=" * 80)
    print("TEXT CHARACTERISTICS ANALYSIS")
    print("=" * 80)
    
    data_dir = "/Users/tod/data/occupations"
    train_df = pd.read_csv(f"{data_dir}/train_df.csv")
    
    # Text length analysis
    text_lengths = train_df['text'].str.len()
    word_counts = train_df['text'].str.split().str.len()
    
    print(f"\n📝 TEXT LENGTH STATISTICS:")
    print(f"   Character length - Min: {text_lengths.min()}, Max: {text_lengths.max()}, Mean: {text_lengths.mean():.1f}")
    print(f"   Word count - Min: {word_counts.min()}, Max: {word_counts.max()}, Mean: {word_counts.mean():.1f}")
    
    # Find examples of very short and very long titles
    print(f"\n📋 EXAMPLE TEXTS:")
    print("   Shortest titles:")
    for i, (idx, row) in enumerate(train_df.loc[text_lengths.nsmallest(5).index].iterrows()):
        print(f"     {i+1}. \"{row['text']}\" -> {row['guide']}")
    
    print("   Longest titles:")
    for i, (idx, row) in enumerate(train_df.loc[text_lengths.nlargest(5).index].iterrows()):
        print(f"     {i+1}. \"{row['text']}\" -> {row['guide']}")


def analyze_problematic_classes():
    """Identify classes that are likely to perform poorly."""
    print("\n" + "=" * 80)
    print("PROBLEMATIC CLASSES ANALYSIS")
    print("=" * 80)
    
    data_dir = "/Users/tod/data/occupations"
    train_df = pd.read_csv(f"{data_dir}/train_df.csv")
    
    class_info = []
    for label in sorted(train_df['label'].unique()):
        class_data = train_df[train_df['label'] == label]
        guide_name = class_data['guide'].iloc[0]
        sample_count = len(class_data)
        
        # Get sample texts for this class
        sample_texts = class_data['text'].tolist()[:3]
        
        class_info.append({
            'label': label,
            'guide': guide_name,
            'count': sample_count,
            'sample_texts': sample_texts
        })
    
    # Sort by sample count (ascending)
    class_info.sort(key=lambda x: x['count'])
    
    print("\n🚨 MOST PROBLEMATIC CLASSES (fewest samples):")
    for i, info in enumerate(class_info[:10]):
        print(f"\n   {i+1}. Label {info['label']}: {info['count']} samples")
        print(f"      Guide: {info['guide']}")
        examples = ', '.join(f'"{t}"' for t in info['sample_texts'])
        print(f"      Examples: {examples}")
    
    print(f"\n📈 BEST REPRESENTED CLASSES:")
    for i, info in enumerate(class_info[-5:]):
        print(f"\n   {i+1}. Label {info['label']}: {info['count']} samples")
        print(f"      Guide: {info['guide']}")
        examples = ', '.join(f'"{t}"' for t in info['sample_texts'][:2])
        print(f"      Examples: {examples}")


def suggest_improvements():
    """Suggest specific improvements based on analysis."""
    print("\n" + "=" * 80)
    print("IMPROVEMENT RECOMMENDATIONS")
    print("=" * 80)
    
    recommendations = [
        "🎯 CLASS IMBALANCE SOLUTIONS:",
        "   • Use weighted random sampling during training",
        "   • Apply class weights in loss function (inverse frequency)",
        "   • Consider data augmentation for underrepresented classes",
        "   • Use focal loss to focus on hard examples",
        "",
        "⚙️ MODEL IMPROVEMENTS:",
        "   • Increase max sequence length (current: 128 -> suggested: 256)",
        "   • Use a more powerful model (bert-large or domain-specific)",
        "   • Increase dropout to prevent overfitting on small classes",
        "   • Lower learning rate for more stable training",
        "",
        "📊 TRAINING STRATEGY:",
        "   • Increase number of epochs with early stopping",
        "   • Use stratified sampling to ensure balanced validation",
        "   • Monitor per-class metrics, not just overall accuracy",
        "   • Consider hierarchical classification (group similar occupations)",
        "",
        "🔍 DATA QUALITY:",
        "   • Review and potentially merge very similar classes",
        "   • Augment data for classes with <10 samples",
        "   • Use external occupation data for training",
        "   • Consider semi-supervised learning approaches"
    ]
    
    for rec in recommendations:
        print(rec)


def main():
    """Run complete analysis of poor model performance."""
    analyze_class_distribution()
    analyze_text_characteristics()
    analyze_problematic_classes()
    suggest_improvements()
    
    print("\n" + "=" * 80)
    print("🚀 NEXT STEPS:")
    print("   1. Run the improved training script:")
    print("      python examples/train_improved_example.py")
    print("   2. Compare results with baseline model")
    print("   3. Consider additional data augmentation if needed")
    print("=" * 80)


if __name__ == "__main__":
    main()