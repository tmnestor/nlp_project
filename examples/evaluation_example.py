"""Example script for evaluating the occupations classifier."""

import os
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from nlp_project.evaluation.evaluate import evaluate_model, print_evaluation_results, save_evaluation_results


def run_evaluation_example():
    """Run evaluation example on test data."""
    print("=" * 60)
    print("OCCUPATIONS CLASSIFIER EVALUATION EXAMPLE")
    print("=" * 60)
    
    model_path = "./models/occupations_classifier"
    test_data_path = "/Users/tod/data/occupations/test_df.csv"
    output_dir = "./evaluation_results"
    
    # Check if model exists
    if not os.path.exists(os.path.join(model_path, "best_model.pt")):
        print(f"❌ Trained model not found at: {model_path}")
        print("Please run the training example first:")
        print("  python examples/train_occupations_example.py")
        return
    
    # Check if test data exists
    if not os.path.exists(test_data_path):
        print(f"❌ Test data not found at: {test_data_path}")
        print("Please ensure the occupations dataset is available.")
        return
    
    print(f"✅ Found trained model: {model_path}")
    print(f"✅ Found test data: {test_data_path}")
    
    print(f"\nEvaluation Parameters:")
    print(f"  Model Path: {model_path}")
    print(f"  Test Data: {test_data_path}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Device: auto")
    
    print("\n" + "=" * 60)
    print("Running evaluation...")
    print("=" * 60)
    
    try:
        # Run evaluation
        results = evaluate_model(
            model_path=model_path,
            test_data_path=test_data_path,
            device="auto"
        )
        
        print("\n✅ Evaluation completed successfully!")
        
        # Print results
        print_evaluation_results(results)
        
        # Save results
        save_evaluation_results(results, output_dir)
        
        print(f"\n📊 Detailed results saved to: {output_dir}")
        
        # Show key metrics summary
        print("\n" + "=" * 60)
        print("KEY METRICS SUMMARY")
        print("=" * 60)
        print(f"Test Accuracy: {results['accuracy']:.4f}")
        print(f"Test Samples: {results['num_test_samples']}")
        print(f"Number of Classes: {len(results['unique_labels'])}")
        print(f"Mean Confidence: {results['confidence_stats']['mean_confidence']:.4f}")
        
        # Show top performing classes
        report = results['classification_report']
        class_f1_scores = []
        for i, label in enumerate(results['unique_labels']):
            label_str = str(label)
            if label_str in report and 'f1-score' in report[label_str]:
                class_name = results['target_names'][i]
                f1_score = report[label_str]['f1-score']
                class_f1_scores.append((class_name, f1_score))
        
        if class_f1_scores:
            class_f1_scores.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n🏆 Top 5 Best Performing Classes (by F1-score):")
            for i, (class_name, f1_score) in enumerate(class_f1_scores[:5], 1):
                print(f"  {i}. {class_name[:40]:40} F1: {f1_score:.4f}")
            
            print(f"\n⚠️  Bottom 5 Classes (by F1-score):")
            for i, (class_name, f1_score) in enumerate(class_f1_scores[-5:], 1):
                print(f"  {i}. {class_name[:40]:40} F1: {f1_score:.4f}")
        
    except Exception as e:
        print(f"❌ Evaluation failed with error: {e}")
        return
    
    print("\n" + "=" * 60)
    print("✅ Evaluation example completed!")
    print("=" * 60)


def run_command_line_example():
    """Show command line usage examples."""
    print("\n" + "=" * 60)
    print("COMMAND LINE EVALUATION EXAMPLES")
    print("=" * 60)
    
    print("\n1. Basic evaluation:")
    print('   python -m nlp_project.evaluation.evaluate \\')
    print('       --model_path ./models/occupations_classifier \\')
    print('       --test_data /Users/tod/data/occupations/test_df.csv')
    
    print("\n2. Evaluation with custom output directory:")
    print('   python -m nlp_project.evaluation.evaluate \\')
    print('       --model_path ./models/occupations_classifier \\')
    print('       --test_data /Users/tod/data/occupations/test_df.csv \\')
    print('       --output_dir ./my_evaluation_results')
    
    print("\n3. Evaluation on specific device:")
    print('   python -m nlp_project.evaluation.evaluate \\')
    print('       --model_path ./models/occupations_classifier \\')
    print('       --test_data /Users/tod/data/occupations/test_df.csv \\')
    print('       --device cpu')
    
    print("\n📁 Output files created:")
    print("   - evaluation_metrics.txt: Detailed metrics and per-class performance")
    print("   - confusion_matrix.csv: Confusion matrix in CSV format")


if __name__ == "__main__":
    run_evaluation_example()
    run_command_line_example()