#!/usr/bin/env python3
"""
LecEval - A toolkit for working with the LecEval multimodal educational presentation dataset.

This script provides a comprehensive command-line interface for analyzing, visualizing,
and evaluating the LecEval dataset containing lecture slides, transcripts, and ratings.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Optional, List
import traceback

# Import LecEval modules
try:
    from utils import setup_logging
    # from utils import setup_logging, get_config
    from dataset import LecEvalDataset
    from analyzer import LecEvalAnalyzer
    from evaluator import LecEvalEvaluator
    from visualizer import LecEvalVisualizer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all required modules are available.")
    sys.exit(1)

# Set up logging
logger = logging.getLogger(__name__)

def validate_paths(args) -> bool:
    """Validate input paths and files."""
    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}")
        return False
    
    if args.images and not os.path.exists(args.images):
        logger.error(f"Images directory not found: {args.images}")
        return False
    
    if args.predictions and not os.path.exists(args.predictions):
        logger.error(f"Predictions file not found: {args.predictions}")
        return False
    
    return True

def setup_output_directory(output_path: Optional[str]) -> Optional[Path]:
    """Set up output directory if specified."""
    if output_path:
        output_dir = Path(output_path)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory: {output_dir}")
            return output_dir
        except Exception as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            return None
    return None

def perform_analysis(dataset: LecEvalDataset, analyzer: LecEvalAnalyzer, 
                    output_dir: Optional[Path], verbose: bool = False) -> None:
    """Perform comprehensive dataset analysis."""
    logger.info("Starting dataset analysis...")
    
    try:
        # Basic statistics
        stats = analyzer.basic_statistics()
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        print(json.dumps(stats, indent=2, default=str))
        
        # Score distributions
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        if output_dir:
            score_dist_path = output_dir / 'score_distributions.png'
            correlation_path = output_dir / 'correlation_heatmap.png'
            overview_path = output_dir / 'dataset_overview.png'
            
            analyzer.plot_score_distributions(str(score_dist_path))
            analyzer.plot_correlation_heatmap(str(correlation_path))
            
            # Create dataset overview with visualizer
            visualizer = LecEvalVisualizer(dataset, analyzer)
            visualizer.create_dataset_overview(str(overview_path))
            
            print(f"✓ Score distributions saved to: {score_dist_path}")
            print(f"✓ Correlation heatmap saved to: {correlation_path}")
            print(f"✓ Dataset overview saved to: {overview_path}")
        else:
            analyzer.plot_score_distributions()
            analyzer.plot_correlation_heatmap()
            
            visualizer = LecEvalVisualizer(dataset, analyzer)
            visualizer.create_dataset_overview()
        
        # Lecture-wise analysis
        try:
            lecture_stats = analyzer.lecture_wise_analysis()
            if not lecture_stats.empty:
                print("\n" + "="*60)
                print("LECTURE-WISE ANALYSIS")
                print("="*60)
                print(lecture_stats.to_string(index=False))
                
                if output_dir:
                    lecture_csv_path = output_dir / 'lecture_statistics.csv'
                    lecture_stats.to_csv(lecture_csv_path, index=False)
                    print(f"✓ Lecture statistics saved to: {lecture_csv_path}")
            else:
                print("\n⚠ No lecture-wise data available for analysis")
        except Exception as e:
            logger.warning(f"Lecture-wise analysis failed: {e}")
            if verbose:
                traceback.print_exc()
        
        # Additional analysis for verbose mode
        if verbose:
            print("\n" + "="*60)
            print("DETAILED ANALYSIS")
            print("="*60)
            
            # Inter-annotator agreement
            try:
                agreement_stats = analyzer.inter_annotator_agreement()
                if agreement_stats:
                    print("\nInter-annotator Agreement:")
                    print(json.dumps(agreement_stats, indent=2, default=str))
            except Exception as e:
                logger.warning(f"Inter-annotator agreement analysis failed: {e}")
            
            # Quality metrics
            try:
                quality_stats = analyzer.quality_metrics()
                if quality_stats:
                    print("\nQuality Metrics:")
                    print(json.dumps(quality_stats, indent=2, default=str))
            except Exception as e:
                logger.warning(f"Quality metrics analysis failed: {e}")
        
        print("\n✓ Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if verbose:
            traceback.print_exc()
        raise

def perform_visualization(dataset: LecEvalDataset, analyzer: LecEvalAnalyzer,
                         sample_ids: List[str], output_dir: Optional[Path],
                         batch_mode: bool = False) -> None:
    """Perform sample visualization."""
    logger.info(f"Starting visualization for {len(sample_ids)} sample(s)...")
    
    visualizer = LecEvalVisualizer(dataset, analyzer)
    
    for sample_id in sample_ids:
        try:
            print(f"\n📊 Visualizing sample: {sample_id}")
            
            output_path = None
            if output_dir:
                output_path = output_dir / f'{sample_id}_analysis.png'
            
            visualizer.create_sample_visualization(sample_id, 
                                                 str(output_path) if output_path else None)
            
            if output_path:
                print(f"✓ Visualization saved to: {output_path}")
            
            if not batch_mode and len(sample_ids) > 1:
                # Pause between visualizations in interactive mode
                input("Press Enter to continue to next sample...")
                
        except Exception as e:
            logger.error(f"Visualization failed for sample {sample_id}: {e}")
            continue
    
    print("\n✓ Visualization completed!")

def perform_evaluation(dataset: LecEvalDataset, predictions_path: str,
                      output_dir: Optional[Path], detailed: bool = False) -> None:
    """Perform model evaluation."""
    logger.info("Starting model evaluation...")
    
    try:
        evaluator = LecEvalEvaluator(dataset)
        
        # Load predictions
        with open(predictions_path, 'r') as f:
            if predictions_path.endswith('.jsonl'):
                predictions = [json.loads(line) for line in f]
            else:
                predictions = json.load(f)
        
        # Evaluate
        # results = evaluator.evaluate_model(predictions)
        # convert
        results = evaluator.evaluate_model(predictions_path, aggregation_method='mean')
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(json.dumps(results, indent=2, default=str))
        
        # Save results
        if output_dir:
            results_path = output_dir / 'evaluation_results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n✓ Results saved to: {results_path}")
        
        # Detailed evaluation
        if detailed:
            try:
                detailed_results = evaluator.detailed_evaluation(predictions)
                print("\n" + "="*60)
                print("DETAILED EVALUATION")
                print("="*60)
                print(json.dumps(detailed_results, indent=2, default=str))
                
                if output_dir:
                    detailed_path = output_dir / 'detailed_evaluation.json'
                    with open(detailed_path, 'w') as f:
                        json.dump(detailed_results, f, indent=2, default=str)
                    print(f"✓ Detailed results saved to: {detailed_path}")
                    
            except Exception as e:
                logger.warning(f"Detailed evaluation failed: {e}")
        
        print("\n✓ Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

def list_samples(dataset: LecEvalDataset, limit: int = 10) -> None:
    """
    List available samples in the dataset.
    """
    print("\n" + "="*60)
    print("AVAILABLE SAMPLES")
    print("="*60)
    
    # Check if the dataset object is empty before processing.
    if not dataset.data:
        print("No samples available in the dataset.")
        return

    # Use dataset.data directly to access the list of samples.
    total = len(dataset.data)
    
    print(f"Total samples: {total}")
    print(f"Showing first {min(limit, total)} samples:\n")
    
    # Iterate over a slice of the dataset to respect the limit.
    for i, sample in enumerate(dataset.data[:limit]):
        sample_id = sample.get('id', f'sample_{i}')
        
        transcript_preview = sample.get('transcript', 'No transcript')[:50]
        if len(transcript_preview) == 50:
            transcript_preview += "..."
        
        print(f"{i+1:3d}. ID: {sample_id}")
        print(f"     Transcript: {transcript_preview}")
        
        if 'rate' in sample:
            rubrics = list(sample['rate'].keys())
            print(f"     Rubrics: {', '.join(rubrics)}")
        print()
    
    if total > limit:
        print(f"... and {total - limit} more samples")

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='LecEval Dataset Utilities - Analyze, visualize, and evaluate educational presentation data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python main.py --data ../dataset/ml-1/metadata.jsonl --action analyze --output ./results

  # Visualize specific samples
  python main.py --data ../dataset/ml-1/metadata.jsonl --action visualize --sample-id ml-1_10_slide_000

  # Batch visualization
  python main.py --data ../dataset/ml-1/metadata.jsonl --action visualize --sample-id ml-1_10_slide_000,ml-1_10_slide_001

  # Evaluate model predictions
  python main.py --data ../dataset/ml-1/metadata.jsonl --action evaluate --predictions predictions.json

  # List available samples
  python main.py --data ../dataset/ml-1/metadata.jsonl --action list --limit 20
        """
    )
    
    # Required arguments
    parser.add_argument('--data', required=True, 
                       help='Path to dataset JSONL file')
    
    # Optional arguments
    parser.add_argument('--images', 
                       help='Path to images directory (optional)')
    parser.add_argument('--action', 
                       choices=['analyze', 'visualize', 'evaluate', 'list'], 
                       default='analyze', 
                       help='Action to perform (default: analyze)')
    parser.add_argument('--sample-id', 
                       help='Sample ID(s) for visualization (comma-separated for multiple)')
    parser.add_argument('--predictions', 
                       help='Path to model predictions JSON/JSONL file')
    parser.add_argument('--output', 
                       help='Output directory for results and plots')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--detailed', action='store_true',
                       help='Enable detailed evaluation (for evaluate action)')
    parser.add_argument('--batch', action='store_true',
                       help='Batch mode for visualizations (no interactive pauses)')
    parser.add_argument('--limit', type=int, default=10,
                       help='Limit for list action (default: 10)')
    parser.add_argument('--log-level', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO',
                       help='Set logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(level=args.log_level)
    
    # Validate paths
    if not validate_paths(args):
        sys.exit(1)
    
    # Set up output directory
    output_dir = setup_output_directory(args.output)
    
    try:
        # Initialize dataset
        logger.info(f"Loading dataset from: {args.data}")
        dataset = LecEvalDataset(args.data, args.images)
        
        # Verify dataset loaded successfully
        if not dataset.data:
            logger.error("Dataset is empty or failed to load")
            sys.exit(1)
        
        logger.info(f"Dataset loaded successfully: {len(dataset.data)} samples")
        
        # Initialize analyzer
        analyzer = LecEvalAnalyzer(dataset)
        
        # Perform requested action
        if args.action == 'analyze':
            perform_analysis(dataset, analyzer, output_dir, args.verbose)
            
        elif args.action == 'visualize':
            if not args.sample_id:
                logger.error("Please provide --sample-id for visualization")
                sys.exit(1)
            
            # Parse sample IDs (support comma-separated list)
            sample_ids = [sid.strip() for sid in args.sample_id.split(',')]
            perform_visualization(dataset, analyzer, sample_ids, output_dir, args.batch)
            
        elif args.action == 'evaluate':
            if not args.predictions:
                logger.error("Please provide --predictions file for evaluation")
                sys.exit(1)
            
            perform_evaluation(dataset, args.predictions, output_dir, args.detailed)
            
        elif args.action == 'list':
            list_samples(dataset, args.limit)
        
        logger.info("Program completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Program failed: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()