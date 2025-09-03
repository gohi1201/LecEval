# LecEval Toolkit

A comprehensive toolkit for analyzing and evaluating multimodal educational presentations using the LecEval dataset and metrics.

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Structure](#data-logical_structure)
- [Components](#components)
  - [Dataset Handler](#dataset-handler)
  - [Analyzer](#analyzer)
  - [Evaluator](#evaluator)
  - [Visualizer](#visualizer)
- [Command Line Interface](#command-line-interface)
- [Tutorials](#tutorials)
  - [Basic Data Analysis](#basic-data-analysis)
  - [Visualization Examples](#visualization-examples)
  - [Model Evaluation](#model-evaluation)
- [Advanced Usage](#advanced-usage)
  - [Custom Analysis](#custom-analysis)
  - [Batch Processing](#batch-processing)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JoylimJY/LecEval
cd LecEval
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Structure

The LecEval dataset uses JSONL format for metadata and organized image directories:

```
LecEval/
├── data/                  # Contains annotation data
│   ├── ml-1/             # Machine Learning lecture series 1
│   ├── psy/              # Psychology lectures
│   └── speaking/         # Public Speaking lectures
└── images/               # Contains slide images
    ├── ml-1/             # ML-1 slides
    ├── psy-1/            # Psychology series 1 slides
    ├── psy-2/            # Psychology series 2 slides
    └── speaking/         # Public Speaking slides
```

### Dataset Format

Each line in `metadata.jsonl` contains a sample with this logical_structure:

```json
{
  "id": "ml-1_10_slide_000",
  "slide": "/images/ml-1/10/slide_000.png",
  "transcript": "Welcome, everyone, to Lecture 5.2 on Alignment and Representation.",
  "rate": {
    "content_relevance": [5, 5, 5],
    "expressive_clarity": [5, 5, 5],
    "logical_structure": [5, 5, 5],
    "audience_engagement": [1, 1, 1]
  },
  "prompt": [
    {
      "role": "user",
      "content": "Instructions:\n\nYou are provided with a segment of a lecture slide..."
    },
    {
      "role": "assistant", 
      "content": "5, 5, 5, 1"
    }
  ]
}
```

### Rubric Criteria

The dataset evaluates presentations on four key rubrics:

- **Content Relevance**: How accurately the transcript represents the slide content
- **Expressive Clarity**: How clear and understandable the language is
- **Logical Structure**: How well the transcript is organized
- **Audience Engagement**: How engaging and inspiring the content is

Each criterion is rated on a scale of 1-5 by multiple annotators.

## Quick Start

Here's a minimal example to get you started:

```python
from dataset import LecEvalDataset
from analyzer import LecEvalAnalyzer
from visualizer import LecEvalVisualizer
from evaluator import LecEvalEvaluator

# Initialize components
dataset = LecEvalDataset(
    data_path="/dataset/ml-1/metadata.jsonl",  # Path to JSONL file
    images_path="/images/ml-1"                      # Base path to images
)
analyzer = LecEvalAnalyzer(dataset)
visualizer = LecEvalVisualizer(dataset, analyzer)

# Get basic statistics
stats = analyzer.basic_statistics()
print(f"Total samples: {stats['total_samples']}")
print(f"Unique lectures: {stats['unique_lectures']}")

# Visualize a sample
visualizer.create_sample_visualization("ml-1_10_slide_000", "output.png")

# Evaluate model predictions
evaluator = LecEvalEvaluator(dataset)
results = evaluator.evaluate_model("predictions.json")
```

## Command Line Interface

The toolkit includes a comprehensive command-line interface:

### Basic Analysis
```bash
# Analyze dataset and generate visualizations
python main.py --data ../dataset/ml-1/metadata.jsonl --images /images --action analyze --output ./results

# Verbose analysis with detailed metrics
python main.py --data ../dataset/ml-1/metadata.jsonl --action analyze --verbose --output ./results
```

### Sample Visualization
```bash
# Visualize a single sample
python main.py --data ../dataset/ml-1/metadata.jsonl --images /images --action visualize --sample-id ml-1_10_slide_000

# Batch visualization of multiple samples
python main.py --data ../dataset/ml-1/metadata.jsonl --images /images --action visualize --sample-id ml-1_10_slide_000,ml-1_10_slide_001,ml-1_11_slide_000 --batch --output ./visualizations
```

### Model Evaluation
```bash
# Evaluate model predictions
python main.py --data /dataset/ml-1/metadata.jsonl --action evaluate --predictions predictions.json --output ./evaluation

# Detailed evaluation with extended metrics
python main.py --data /dataset/ml-1/metadata.jsonl --action evaluate --predictions predictions.jsonl --detailed --output ./evaluation
```

### Dataset Exploration
```bash
# List available samples
python main.py --data /dataset/ml-1/metadata.jsonl --action list --limit 20
```

## Components

### Dataset Handler

The `LecEvalDataset` class manages JSONL dataset loading and operations:

```python
# Initialize dataset
dataset = LecEvalDataset(
    data_path="/dataset/ml-1/metadata.jsonl",
    images_path="/images/ml-1"
)

# Get a specific sample
sample = dataset.get_sample_by_id("ml-1_10_slide_000")
print(f"Transcript: {sample['transcript']}")
print(f"Content_Relevance scores: {sample['rate']['content_relevance']}")

# Load slide image
image = dataset.get_image(sample)

# Get all samples
all_samples = dataset.get_all_samples()
print(f"Total samples: {len(all_samples)}")
```

### Analyzer

The `LecEvalAnalyzer` class provides statistical analysis tools:

```python
analyzer = LecEvalAnalyzer(dataset)

# Get comprehensive statistics
stats = analyzer.basic_statistics()
print(f"Dataset contains {stats['total_samples']} samples")

# Analyze rubric correlations
correlations = analyzer.correlation_analysis()
print("Rubric correlations:")
for rubric_pair, correlation in correlations.items():
    print(f"  {rubric_pair}: {correlation:.3f}")

# Analyze lecture-wise performance
lecture_stats = analyzer.lecture_wise_analysis()
print("\nLecture performance:")
print(lecture_stats.head())

# Inter-annotator agreement
agreement = analyzer.inter_annotator_agreement()
print(f"Average inter-annotator agreement: {agreement['overall_agreement']:.3f}")
```

### Evaluator

The `LecEvalEvaluator` class handles model evaluation with multiple metrics:

```python
evaluator = LecEvalEvaluator(dataset)

# Prepare predictions in the correct format
predictions = [
    {
        "id": "ml-1_10_slide_000",
        "predictions": {
            "content_relevance": 4.5,
            "expressive_clarity": 4.8,
            "logical_structure": 4.2,
            "audience_engagement": 2.1
        }
    },
    # ... more predictions
]

# Evaluate predictions
results = evaluator.evaluate_model(predictions)
print("Evaluation Results:")
for rubric in ['content_relevance', 'expressive_clarity', 'logical_structure', 'audience_engagement']:
    metrics = results[rubric]
    print(f"\n{rubric.title()}:")
    print(f"  Spearman ρ: {metrics['spearman_r']:.3f} (p={metrics['spearman_p']:.3f})")
    print(f"  Pearson r: {metrics['pearson_r']:.3f} (p={metrics['pearson_p']:.3f})")
    print(f"  MSE: {metrics['mse']:.3f}")
    print(f"  MAE: {metrics['mae']:.3f}")
```

### Visualizer

The `LecEvalVisualizer` class creates comprehensive visualizations:

```python
visualizer = LecEvalVisualizer(dataset, analyzer)

# Create detailed sample visualization
visualizer.create_sample_visualization(
    "ml-1_10_slide_000",
    "sample_analysis.png"
)

# Create dataset overview
visualizer.create_dataset_overview("dataset_overview.png")
```

## Tutorials

### Basic Data Analysis

Here's how to perform comprehensive analysis on the dataset:

```python
# Initialize components
dataset = LecEvalDataset("/dataset/ml-1/metadata.jsonl", "/images/ml-1")
analyzer = LecEvalAnalyzer(dataset)

# Get overall statistics
stats = analyzer.basic_statistics()
print("\nDataset Overview:")
print(f"Total samples: {stats['total_samples']}")
print(f"Unique lectures: {stats['unique_lectures']}")

# Print rubric statistics
rubrics = ['content_relevance', 'expressive_clarity', 'logical_structure', 'audience_engagement']
for rubric in rubrics:
    metrics = stats['rubric_statistics'][rubric]
    print(f"\n{rubric.title()}:")
    print(f"  Mean: {metrics['mean']:.2f}")
    print(f"  Std: {metrics['std']:.2f}")
    print(f"  Min: {metrics['min']:.2f}")
    print(f"  Max: {metrics['max']:.2f}")

# Analyze score distributions
print("\nScore Distribution Analysis:")
for rubric in rubrics:
    scores = []
    for sample in dataset.data:
        if 'rate' in sample and rubric in sample['rate']:
            scores.extend(sample['rate'][rubric])
    
    if scores:
        print(f"{rubric.title()}: {len(scores)} total ratings")
        unique_scores = list(set(scores))
        for score in sorted(unique_scores):
            count = scores.count(score)
            percentage = (count / len(scores)) * 100
            print(f"  Score {score}: {count} ({percentage:.1f}%)")
```

### Visualization Examples

Create insightful visualizations for different analysis needs:

```python
# Initialize visualizer
visualizer = LecEvalVisualizer(dataset, analyzer)

def analyze_lecture_series(lecture_prefix: str):
    """Analyze all slides from a specific lecture."""
    lecture_samples = [
        sample for sample in dataset.data 
        if sample['id'].startswith(lecture_prefix)
    ]
    
    print(f"\nAnalyzing {lecture_prefix} ({len(lecture_samples)} slides):")
    
    # Calculate average scores per rubric
    rubric_averages = {}
    for rubric in ['content_relevance', 'expressive_clarity', 'logical_structure', 'audience_engagement']:
        all_scores = []
        for sample in lecture_samples:
            if 'rate' in sample and rubric in sample['rate']:
                all_scores.extend(sample['rate'][rubric])
        
        if all_scores:
            rubric_averages[rubric] = {
                'mean': sum(all_scores) / len(all_scores),
                'count': len(all_scores)
            }
    
    # Display results
    for rubric, stats in rubric_averages.items():
        print(f"  {rubric.title()}: {stats['mean']:.2f} (n={stats['count']})")
    
    return rubric_averages

# Analyze specific lectures
analyze_lecture_series("ml-1_10")
analyze_lecture_series("ml-1_11")

# Create visualizations for high-performing and low-performing samples
def find_extreme_samples():
    """Find samples with highest and lowest overall scores."""
    sample_scores = []
    
    for sample in dataset.data:
        if 'rate' not in sample:
            continue
        
        total_score = 0
        count = 0
        for rubric_scores in sample['rate'].values():
            total_score += sum(rubric_scores)
            count += len(rubric_scores)
        
        if count > 0:
            avg_score = total_score / count
            sample_scores.append((sample['id'], avg_score))
    
    # Sort by average score
    sample_scores.sort(key=lambda x: x[1])
    
    print("\nLowest scoring samples:")
    for sample_id, score in sample_scores[:3]:
        print(f"  {sample_id}: {score:.2f}")
        # Create visualization
        visualizer.create_sample_visualization(
            sample_id, 
            f"low_score_{sample_id}.png"
        )
    
    print("\nHighest scoring samples:")
    for sample_id, score in sample_scores[-3:]:
        print(f"  {sample_id}: {score:.2f}")
        # Create visualization
        visualizer.create_sample_visualization(
            sample_id, 
            f"high_score_{sample_id}.png"
        )

find_extreme_samples()
```

### Model Evaluation

Comprehensive model evaluation workflow:

```python
# Initialize evaluator
evaluator = LecEvalEvaluator(dataset)

def create_sample_predictions():
    """Create sample predictions for demonstration."""
    predictions = []
    
    for sample in dataset.data[:10]:  # First 10 samples
        # Simulate model predictions (replace with actual model output)
        sample_pred = {
            "id": sample['id'],
            "predictions": {
                "content_relevance": min(5, max(1, 
                    sum(sample['rate']['content_relevance']) / len(sample['rate']['content_relevance']) + 
                    (hash(sample['id']) % 3 - 1) * 0.5  # Add some noise
                )),
                "expressive_clarity": min(5, max(1,
                    sum(sample['rate']['expressive_clarity']) / len(sample['rate']['expressive_clarity']) +
                    (hash(sample['id']) % 5 - 2) * 0.3
                )),
                "logical_structure": min(5, max(1,
                    sum(sample['rate']['logical_structure']) / len(sample['rate']['logical_structure']) +
                    (hash(sample['id']) % 7 - 3) * 0.4
                )),
                "audience_engagement": min(5, max(1,
                    sum(sample['rate']['audience_engagement']) / len(sample['rate']['audience_engagement']) +
                    (hash(sample['id']) % 9 - 4) * 0.6
                ))
            }
        }
        predictions.append(sample_pred)
    
    return predictions

def detailed_evaluation_analysis(predictions):
    """Perform detailed evaluation analysis."""
    results = evaluator.evaluate_model(predictions)
    
    print("Detailed Evaluation Results:")
    print("=" * 50)
    
    rubrics = ['content_relevance', 'expressive_clarity', 'logical_structure', 'audience_engagement']
    
    # Overall performance summary
    overall_correlations = []
    for rubric in rubrics:
        spearman_r = results[rubric]['spearman_r']
        overall_correlations.append(spearman_r)
        
    print(f"Overall Performance (Spearman ρ): {sum(overall_correlations)/len(overall_correlations):.3f}")
    print()
    
    # Per-rubric detailed analysis
    for rubric in rubrics:
        metrics = results[rubric]
        print(f"{rubric.upper()} Analysis:")
        print(f"  Correlation (Spearman): {metrics['spearman_r']:.3f} (p={metrics['spearman_p']:.3f})")
        print(f"  Correlation (Pearson):  {metrics['pearson_r']:.3f} (p={metrics['pearson_p']:.3f})")
        print(f"  Mean Squared Error:     {metrics['mse']:.3f}")
        print(f"  Mean Absolute Error:    {metrics['mae']:.3f}")
        
        # Interpretation
        if metrics['spearman_r'] > 0.7:
            performance = "Excellent"
        elif metrics['spearman_r'] > 0.5:
            performance = "Good"
        elif metrics['spearman_r'] > 0.3:
            performance = "Moderate"
        else:
            performance = "Poor"
        print(f"  Performance Level:      {performance}")
        print()
    
    return results

# Generate and evaluate predictions
sample_predictions = create_sample_predictions()
evaluation_results = detailed_evaluation_analysis(sample_predictions)

# Save results
import json
with open("sample_evaluation_results.json", "w") as f:
    json.dump(evaluation_results, f, indent=2, default=str)
print("Results saved to sample_evaluation_results.json")
```

## Advanced Usage

### Custom Analysis

Extend the analyzer for domain-specific metrics:

```python
import re
from collections import Counter

class CustomLecEvalAnalyzer(LecEvalAnalyzer):
    def analyze_transcript_complexity(self):
        """Analyze linguistic complexity of transcripts."""
        complexity_metrics = []
        
        for sample in self.dataset.data:
            transcript = sample.get('transcript', '')
            
            # Basic metrics
            words = transcript.split()
            sentences = re.split(r'[.!?]+', transcript)
            
            if len(words) > 0 and len(sentences) > 0:
                metrics = {
                    'id': sample['id'],
                    'word_count': len(words),
                    'sentence_count': len([s for s in sentences if s.strip()]),
                    'avg_word_length': sum(len(w) for w in words) / len(words),
                    'avg_sentence_length': len(words) / len([s for s in sentences if s.strip()]),
                    'unique_words': len(set(word.lower() for word in words)),
                    'lexical_diversity': len(set(word.lower() for word in words)) / len(words)
                }
                
                # Educational vocabulary analysis
                edu_keywords = [
                    'concept', 'theory', 'principle', 'methodology', 'analysis',
                    'framework', 'approach', 'technique', 'strategy', 'implementation'
                ]
                edu_word_count = sum(1 for word in words 
                                   if word.lower() in edu_keywords)
                metrics['educational_vocabulary_ratio'] = edu_word_count / len(words)
                
                complexity_metrics.append(metrics)
        
        return pd.DataFrame(complexity_metrics)
    
    def analyze_rubric_relationships(self):
        """Analyze relationships between rubrics and transcript features."""
        import scipy.stats as stats
        
        complexity_df = self.analyze_transcript_complexity()
        relationships = {}
        
        rubrics = ['content_relevance', 'expressive_clarity', 'logical_structure', 'audience_engagement']
        complexity_features = [
            'word_count', 'avg_word_length', 'lexical_diversity', 
            'educational_vocabulary_ratio'
        ]
        
        for rubric in rubrics:
            rubric_scores = []
            feature_values = {feature: [] for feature in complexity_features}
            
            for _, row in complexity_df.iterrows():
                sample_id = row['id']
                sample = self.dataset.get_sample_by_id(sample_id)
                
                if sample and 'rate' in sample and rubric in sample['rate']:
                    avg_score = sum(sample['rate'][rubric]) / len(sample['rate'][rubric])
                    rubric_scores.append(avg_score)
                    
                    for feature in complexity_features:
                        feature_values[feature].append(row[feature])
            
            # Calculate correlations
            relationships[rubric] = {}
            for feature in complexity_features:
                if len(rubric_scores) > 1 and len(feature_values[feature]) > 1:
                    correlation, p_value = stats.spearmanr(rubric_scores, feature_values[feature])
                    relationships[rubric][feature] = {
                        'correlation': correlation,
                        'p_value': p_value
                    }
        
        return relationships

# Usage example
custom_analyzer = CustomLecEvalAnalyzer(dataset)

# Analyze transcript complexity
complexity_analysis = custom_analyzer.analyze_transcript_complexity()
print("Transcript Complexity Analysis:")
print(complexity_analysis.describe())

# Analyze rubric-complexity relationships
relationships = custom_analyzer.analyze_rubric_relationships()
print("\nRubric-Complexity Relationships:")
for rubric, features in relationships.items():
    print(f"\n{rubric.title()}:")
    for feature, stats in features.items():
        significance = "***" if stats['p_value'] < 0.001 else "**" if stats['p_value'] < 0.01 else "*" if stats['p_value'] < 0.05 else ""
        print(f"  {feature}: r={stats['correlation']:.3f} (p={stats['p_value']:.3f}){significance}")
```

### Batch Processing

Process multiple lecture series efficiently:

```python
def batch_analyze_lectures(base_path: str, lecture_series: List[str]):
    """Analyze multiple lecture series in batch."""
    results = {}
    
    for series in lecture_series:
        print(f"\nProcessing {series}...")
        
        try:
            # Initialize dataset for this series
            dataset_path = f"{base_path}/dataset/{series}/metadata.jsonl"
            images_path = f"{base_path}/images"
            
            dataset = LecEvalDataset(dataset_path, images_path)
            analyzer = LecEvalAnalyzer(dataset)
            
            # Perform analysis
            stats = analyzer.basic_statistics()
            
            # Extract key metrics
            series_results = {
                'total_samples': stats['total_samples'],
                'rubric_means': {},
                'rubric_stds': {}
            }
            
            for rubric in ['content_relevance', 'expressive_clarity', 'logical_structure', 'audience_engagement']:
                if rubric in stats['rubric_statistics']:
                    series_results['rubric_means'][rubric] = stats['rubric_statistics'][rubric]['mean']
                    series_results['rubric_stds'][rubric] = stats['rubric_statistics'][rubric]['std']
            
            results[series] = series_results
            print(f"✓ {series}: {stats['total_samples']} samples processed")
            
        except Exception as e:
            print(f"✗ Error processing {series}: {e}")
            results[series] = {'error': str(e)}
    
    return results

def compare_lecture_series(results: dict):
    """Compare results across lecture series."""
    print("\n" + "="*60)
    print("LECTURE SERIES COMPARISON")
    print("="*60)
    
    # Create comparison table
    comparison_data = []
    rubrics = ['content_relevance', 'expressive_clarity', 'logical_structure', 'audience_engagement']
    
    for series, data in results.items():
        if 'error' not in data:
            row = {'Series': series, 'Samples': data['total_samples']}
            for rubric in rubrics:
                if rubric in data['rubric_means']:
                    row[f'{rubric.title()}_Mean'] = f"{data['rubric_means'][rubric]:.2f}"
                    row[f'{rubric.title()}_Std'] = f"{data['rubric_stds'][rubric]:.2f}"
            comparison_data.append(row)
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
    
    return comparison_data

# Example usage
lecture_series = ['ml-1']  # Add more series as available
batch_results = batch_analyze_lectures("/path/to/leceval", lecture_series)
comparison_table = compare_lecture_series(batch_results)
```
