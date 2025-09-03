from dataset import LecEvalDataset
from analyzer import LecEvalAnalyzer
from visualizer import LecEvalVisualizer
from evaluator import LecEvalEvaluator

# Initialize components
dataset = LecEvalDataset(
    data_path="../dataset/ml-1/metadata.jsonl",  # Path to JSONL file
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