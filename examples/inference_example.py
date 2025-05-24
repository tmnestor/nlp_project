"""Example script for occupation classification inference."""

import os
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))  # noqa: E402

from nlp_project.inference.predict import OccupationClassifier  # noqa: E402


def run_inference_examples():
    """Run inference examples with sample occupation titles."""
    print("=" * 60)
    print("OCCUPATIONS CLASSIFIER INFERENCE EXAMPLE")
    print("=" * 60)

    model_path = "./models/occupations_classifier"

    # Check if model exists
    if not os.path.exists(os.path.join(model_path, "best_model.pt")):
        print(f"❌ Trained model not found at: {model_path}")
        print("Please run the training example first:")
        print("  python examples/train_occupations_example.py")
        return

    print(f"✅ Found trained model: {model_path}")

    # Initialize classifier
    print("\nInitializing classifier...")
    try:
        classifier = OccupationClassifier(model_path)
        print("✅ Classifier loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load classifier: {e}")
        return

    # Sample occupation titles to classify
    sample_occupations = [
        "Software Engineer",
        "Medical Doctor",
        "Elementary School Teacher",
        "Criminal Defense Lawyer",
        "Construction Worker",
        "Registered Nurse",
        "Data Scientist",
        "Police Officer",
        "Chef",
        "Airline Pilot",
        "Firefighter",
        "Social Worker",
        "Accountant",
        "Graphic Designer",
        "Electrician"
    ]

    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS")
    print("=" * 60)

    # Run predictions
    for occupation in sample_occupations:
        try:
            result = classifier.predict(occupation, return_probs=True)

            print(f"\n📝 Input: {result['input_text']}")
            print(f"🎯 Predicted Category: {result['predicted_guide']}")
            print(f"📊 Confidence: {result['confidence']:.4f}")

            print("   Top 3 Predictions:")
            for pred in result['top_predictions'][:3]:
                print(f"     {pred['rank']}. {pred['guide']} ({pred['probability']:.4f})")

        except Exception as e:
            print(f"❌ Error predicting '{occupation}': {e}")

    print("\n" + "=" * 60)
    print("BATCH PREDICTION EXAMPLE")
    print("=" * 60)

    # Batch prediction example
    batch_occupations = [
        "Python Developer",
        "Heart Surgeon",
        "High School Math Teacher"
    ]

    try:
        batch_results = classifier.predict_batch(batch_occupations)

        print(f"\nProcessed {len(batch_results)} occupations in batch:")
        for i, result in enumerate(batch_results, 1):
            print(f"  {i}. {result['input_text']} → {result['predicted_guide']} "
                  f"(confidence: {result['confidence']:.4f})")

    except Exception as e:
        print(f"❌ Error in batch prediction: {e}")

    print("\n" + "=" * 60)
    print("✅ Inference examples completed!")
    print("=" * 60)


def run_command_line_example():
    """Show command line usage examples."""
    print("\n" + "=" * 60)
    print("COMMAND LINE USAGE EXAMPLES")
    print("=" * 60)

    print("\n1. Single prediction:")
    print('   python -m nlp_project.inference.predict \\')
    print('       --model_path ./models/occupations_classifier \\')
    print('       --text "Software Engineer"')

    print("\n2. Single prediction with probabilities:")
    print('   python -m nlp_project.inference.predict \\')
    print('       --model_path ./models/occupations_classifier \\')
    print('       --text "Medical Doctor" \\')
    print('       --return_probs')

    print("\n3. Batch prediction from file:")
    print('   echo -e "Data Scientist\\nTeacher\\nLawyer" > occupations.txt')
    print('   python -m nlp_project.inference.predict \\')
    print('       --model_path ./models/occupations_classifier \\')
    print('       --file occupations.txt')

    print("\n4. Using specific device:")
    print('   python -m nlp_project.inference.predict \\')
    print('       --model_path ./models/occupations_classifier \\')
    print('       --text "Engineer" \\')
    print('       --device cpu')


if __name__ == "__main__":
    run_inference_examples()
    run_command_line_example()
