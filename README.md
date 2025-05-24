# NLP Project - Occupations Classification

A PyTorch-based text classification system for categorizing occupation titles into professional guides/categories using **advanced triplet loss learning** with BERT.

## 🎯 Overview

This project implements a state-of-the-art machine learning pipeline for classifying occupation titles into categories. It uses **triplet loss learning** combined with BERT to learn semantic embeddings that understand occupational similarity, enabling superior classification performance on imbalanced datasets.

## 📁 Project Structure

```
nlp_project/
├── src/nlp_project/
│   ├── data/
│   │   ├── dataset.py              # Base dataset class
│   │   └── occupations_loader.py   # Occupations-specific data loading
│   ├── models/
│   │   ├── triplet_classifier.py   # Triplet loss BERT classifier (primary)
│   │   └── classifier.py           # Standard BERT classifier (legacy)
│   ├── training/
│   │   ├── train_triplet_occupations.py  # Triplet training (primary)
│   │   └── train_occupations*.py   # Legacy training scripts
│   ├── inference/
│   │   ├── triplet_predict.py      # Triplet inference (primary)
│   │   └── predict.py              # Legacy inference
│   └── evaluation/
│       └── evaluate.py             # Evaluation script
├── tests/                          # Unit tests
├── examples/                       # Example usage scripts
├── scripts/                        # Analysis and utility scripts
├── environment-dev.yml             # Conda environment file
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Environment Setup

Create and activate the conda environment:

```bash
# Create environment from file
mamba env create -f environment-dev.yml

# Activate environment
conda activate nlp_project_dev
```

**Note**: The environment files pin NumPy to version 1.x (`numpy<2`) to avoid compatibility issues with compiled modules. If you encounter NumPy 2.x related errors, update your environment:

```bash
# Update existing environment
mamba env update -f environment-dev.yml --prune

# Or recreate environment
mamba env remove -n nlp_project_dev
mamba env create -f environment-dev.yml
```

### 2. Dataset

The project uses the occupations dataset located at `/Users/tod/data/occupations/` with the following files:
- `train_df.csv` - Training data (text, label, guide)
- `val_df.csv` - Validation data
- `test_df.csv` - Test data

Each row contains:
- `text`: Occupation title (e.g., "Software Engineer")
- `label`: Numeric category ID
- `guide`: Category name (e.g., "Information Technology Workers")

### 3. Training

Train the model using **triplet loss** (recommended):

```bash
# Standard triplet training
python examples/train_triplet_example.py

# Or with custom parameters
python -m nlp_project.training.train_triplet_occupations \
    --data_dir /Users/tod/data/occupations \
    --model_name bert-base-uncased \
    --batch_size 32 \
    --epochs 15 \
    --embedding_dim 256 \
    --triplet_margin 0.5 \
    --output_dir ./models/occupations_triplet_classifier
```

**Why Triplet Loss?**
- 🎯 Learns semantic similarity between occupations
- 📊 Superior performance on imbalanced datasets  
- 🔍 Enables similarity search capabilities
- 🧠 Rich 256D embeddings for better representations

### 4. Inference

Make predictions using the **triplet model**:

```bash
# Basic prediction
python -m nlp_project.inference.triplet_predict \
    --model_path ./models/occupations_triplet_classifier \
    --text "Machine Learning Engineer" \
    --return_probs

# Find similar occupations (unique capability)
echo -e "Software Engineer\nData Scientist\nTeacher\nChef\nNurse" > occupations.txt
python -m nlp_project.inference.triplet_predict \
    --model_path ./models/occupations_triplet_classifier \
    --find_similar "AI Researcher" \
    --reference_file occupations.txt

# Batch prediction with embeddings
python -m nlp_project.inference.triplet_predict \
    --model_path ./models/occupations_triplet_classifier \
    --file occupations.txt \
    --return_embeddings
```

**Triplet Model Advantages:**
- 🔍 **Similarity search**: Find occupations similar to a query
- 🧠 **Rich embeddings**: 256D semantic representations
- 📊 **Better accuracy**: Especially on rare occupation types
- 🎯 **Confidence scores**: More reliable predictions

### 5. Evaluation

Evaluate the triplet model performance:

```bash
python -m nlp_project.evaluation.evaluate \
    --model_path ./models/occupations_triplet_classifier \
    --test_data /Users/tod/data/occupations/test_df.csv \
    --output_dir ./evaluation_results
```

**Expected Performance:**
- 📈 **Macro F1**: 0.6+ (vs 0.38 baseline)
- 🎯 **Rare classes**: Significant improvement on <10 sample classes
- 🧠 **Embeddings**: Meaningful similarity relationships learned

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## 🔧 Code Quality

The project follows strict code quality standards:

```bash
# Linting and formatting
ruff check src/
ruff format src/

# Type checking
mypy src/
```

## 📊 Triplet Loss Architecture

### Overview

The triplet loss approach learns semantic embeddings where similar occupations are close in vector space while dissimilar ones are far apart. This enables both accurate classification and meaningful similarity search.

**Architecture Components:**
- **Base Model**: BERT (bert-base-uncased) for text encoding
- **Encoder**: Shared TripletTextEncoder with dropout regularization
- **Classifier Head**: Linear layer for category prediction
- **Embeddings**: 256-dimensional L2-normalized representations
- **Input**: Occupation title (max 256 tokens)
- **Output**: Classifications + semantic embeddings

### Loss Function

The model uses a **combined loss function** that optimizes two objectives simultaneously:

```
L_total = α * L_classification + β * L_triplet

Where:
- L_classification = CrossEntropyLoss(logits, labels)
- L_triplet = TripletMarginLoss(anchor, positive, negative, margin=0.5)
- α = classification_weight (default: 1.0)
- β = triplet_weight (default: 0.1)  # Reduced to prevent overwhelming classification
```

**Triplet Loss Mathematical Definition:**
```
L_triplet = max(0, d(a,p) - d(a,n) + margin)

Where:
- d(a,p) = distance between anchor and positive (same class)
- d(a,n) = distance between anchor and negative (different class)  
- margin = minimum separation between positive and negative (0.5)
```

### Hard Negative Mining

The system implements **hard negative mining** for more effective learning:

1. **Compute similarity matrix** for all embeddings in batch
2. **Find hardest negatives**: Most similar embeddings from different classes
3. **Mine challenging triplets**: (anchor, positive, hard_negative)
4. **Progressive difficulty**: Starts random mining, switches to hard mining after epoch 2

### Dataset Requirements for Triplet Training

**Class Distribution Analysis:**
- **Total samples**: ~50,000 occupations
- **Classes**: 38 professional categories
- **Imbalance ratio**: 99.5:1 (severe)
- **Rare classes**: 11 classes with <10 samples
- **Critical challenge**: 3 classes with only 2-3 samples

**Triplet Mining Strategy:**
```python
# Requires minimum batch size for effective triplet formation
batch_size >= 32  # Ensures multiple classes per batch
triplet_margin = 0.5  # Optimal separation distance
embedding_dim = 256  # Rich representation space
```

**Data Preprocessing:**
- **Weighted sampling**: Balances rare classes during training
- **Class weights**: Applied to classification loss
- **Maximum length**: 256 tokens for longer job descriptions
- **Tokenization**: BERT WordPiece tokenization

### Training Configuration

**Optimizer Setup:**
```python
optimizer = AdamW(
    model.parameters(), 
    lr=2e-5,  # Optimal learning rate for BERT fine-tuning with triplet loss
    weight_decay=0.01  # L2 regularization
)

scheduler = LinearScheduleWithWarmup(
    optimizer,
    num_warmup_steps=0.1 * total_steps,
    num_training_steps=total_steps
)
```

**Training Strategy:**
- **Epoch 1**: Classification-only training for stable initialization
- **Epochs 2-3**: Random triplet mining with reduced triplet weight (0.1)
- **Epochs 4+**: Hard negative mining for challenging learning
- **Early stopping**: Patience of 4 epochs
- **Gradient clipping**: Max norm 1.0 for stability

### Advantages Over Standard Classification

**1. Semantic Understanding**
- Learns occupational similarity beyond category labels
- Enables "Software Engineer" to be close to "Data Engineer"
- Discovers hidden relationships in professional domains

**2. Improved Performance on Imbalanced Data**
- Better representation learning for rare occupations
- Triplet constraints prevent mode collapse on majority classes
- Hard negative mining focuses on difficult distinctions

**3. Similarity Search Capabilities**
- **Cosine similarity**: Find occupations similar to query
- **Embedding visualization**: t-SNE/UMAP clustering analysis
- **Career path analysis**: Related occupation recommendations

**4. Transfer Learning Benefits**
- **Rich embeddings**: 256D semantic representations
- **Domain adaptation**: Embeddings useful for related tasks
- **Few-shot learning**: Better performance on new occupation types

### References

**Foundational Papers:**
1. **Schroff et al. (2015)**: "FaceNet: A Unified Embedding for Face Recognition and Clustering" - Original triplet loss formulation
2. **Hermans et al. (2017)**: "In Defense of the Triplet Loss for Person Re-Identification" - Hard negative mining strategies
3. **Devlin et al. (2018)**: "BERT: Pre-training of Deep Bidirectional Transformers" - Base transformer architecture

**Technical Innovations:**
- **Combined loss**: Simultaneous classification and embedding learning
- **Progressive mining**: Curriculum learning from easy to hard triplets
- **Class-aware sampling**: Addresses severe imbalance in occupational data
- **L2 normalization**: Ensures unit sphere embedding space for stable cosine similarity

## 🎯 Example Results

```
Input: Software Engineer
Predicted Category: Information Technology Workers
Confidence: 0.9234

Top 3 Predictions:
  1. Information Technology Workers (0.9234)
  2. Engineering Professionals (0.0543)
  3. Office Workers (0.0123)
```

## 📈 Performance Metrics

The evaluation script provides comprehensive metrics:
- **Accuracy**: Overall classification accuracy
- **Per-class Metrics**: Precision, recall, F1-score for each category
- **Confusion Matrix**: Detailed classification breakdown
- **Confidence Statistics**: Model confidence analysis

## 🛠️ API Usage

You can also use the classifier programmatically:

```python
from nlp_project.inference.triplet_predict import TripletOccupationClassifier

# Initialize triplet classifier
classifier = TripletOccupationClassifier("./models/occupations_triplet_classifier")

# Single prediction with embeddings
result = classifier.predict("Data Scientist", return_probs=True, return_embeddings=True)
print(f"Category: {result['predicted_guide']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Embedding shape: {result['embeddings'].shape}")

# Similarity search (unique to triplet models)
reference_jobs = ["Software Engineer", "Research Scientist", "Product Manager"]
similar = classifier.find_similar_occupations("AI Engineer", reference_jobs, top_k=3)
for item in similar:
    print(f"{item['rank']}. {item['text']} (similarity: {item['similarity']:.3f})")

# Batch predictions
occupations = ["Teacher", "Lawyer", "Engineer"]
results = classifier.predict_batch(occupations)
```

## 🔄 Triplet Training Pipeline

1. **Data Loading & Balancing**: Load occupations dataset with weighted sampling for class balance
2. **Tokenization**: Convert occupation titles to BERT tokens (max 256 length)
3. **Triplet Formation**: Mine anchor-positive-negative triplets from each batch
4. **Combined Training**: Optimize both triplet loss and classification loss simultaneously
5. **Progressive Mining**: Start with random triplets, progress to hard negative mining
6. **Validation**: Monitor classification accuracy and embedding quality
7. **Model Saving**: Save best model with embeddings, mappings, and training config

## 📋 Requirements

- Python 3.11+
- PyTorch with MPS support (Apple Silicon) or CUDA (GPU)
- Transformers library
- Standard ML libraries (scikit-learn, pandas, numpy)

See `environment-dev.yml` for complete dependency list.

## ⚠️ Known Issues & Solutions

### Poor Baseline Performance (F1: 0.38)

The initial model shows poor performance due to **severe class imbalance**:
- 38 classes with 99.5:1 imbalance ratio
- 11 classes have <10 samples
- 3 classes have only 2-3 samples

**Solution**: Use the improved training script:
```bash
python examples/train_improved_example.py
```

**Improvements implemented**:
- ✅ **Weighted sampling**: Balances training data
- ✅ **Class-weighted loss**: Penalizes errors on rare classes  
- ✅ **Increased sequence length**: 128→256 for longer job titles
- ✅ **Lower learning rate**: 2e-5→1e-5 for stable training
- ✅ **Higher dropout**: 0.1→0.3 to prevent overfitting
- ✅ **More epochs**: 3→10 with early stopping
- ✅ **L2 regularization**: Weight decay 0.01

**Advanced: Triplet Loss Training**:
- 🔺 **Triplet Loss**: Learns semantic similarity between occupations
- 🎯 **Hard negative mining**: Finds challenging examples for better learning
- 📐 **256D embeddings**: Rich representations for similarity search
- 🔄 **Combined loss**: Triplet + classification for dual objectives
- 📊 **Larger batches**: 32 samples for effective triplet mining

### Analysis Tools

Analyze model performance issues:
```bash
python scripts/analyze_poor_results.py
```

## 🔧 Technical Troubleshooting

### NumPy Compatibility Issues

```bash
# Update environment to use NumPy 1.x
mamba env update -f environment-dev.yml --prune
```

### CUDA/MPS Device Issues

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
```

### Memory Issues

- Reduce batch size: `--batch_size 8`
- Use gradient accumulation  
- Enable mixed precision training

## 🤝 Contributing

1. Follow the existing code style (ruff formatting)
2. Add type hints to all functions
3. Write tests for new functionality
4. Update documentation as needed

## 📝 License

This project is for educational and research purposes.