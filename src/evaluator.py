"""
LecEval evaluation module for model predictions
"""

from utils import *
from dataset import LecEvalDataset

class LecEvalEvaluator:
    """Class for evaluating model predictions against LecEval ground truth."""
    
    def __init__(self, dataset: LecEvalDataset):
        self.dataset = dataset
        self.rubrics = self._get_available_rubrics()
    
    def _get_available_rubrics(self) -> List[str]:
        """Get list of available rubrics from the dataset."""
        rubrics = set()
        for sample in self.dataset.data:
            rate_data = sample.get('rate', {})
            rubrics.update(rate_data.keys())
        return list(rubrics)
    
    def calculate_correlations(self, predictions: Dict[str, List[float]], 
                             ground_truth: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate Spearman and Pearson correlations between predictions and ground truth.
        
        Args:
            predictions: Dict with rubric names as keys and prediction lists as values
            ground_truth: Dict with rubric names as keys and ground truth lists as values
        
        Returns:
            Dict containing correlation scores for each rubric
        """
        correlations = {}
        
        for rubric in self.rubrics:
            if rubric in predictions and rubric in ground_truth:
                pred_scores = predictions[rubric]
                true_scores = ground_truth[rubric]
                
                if len(pred_scores) != len(true_scores):
                    logger.warning(f"Length mismatch for {rubric}: {len(pred_scores)} vs {len(true_scores)}")
                    continue
                
                if len(pred_scores) < 2:
                    logger.warning(f"Insufficient data points for {rubric}: {len(pred_scores)}")
                    continue
                
                try:
                    # Calculate correlations
                    spearman_corr, spearman_p = spearmanr(pred_scores, true_scores)
                    pearson_corr, pearson_p = pearsonr(pred_scores, true_scores)
                    
                    # Handle NaN values
                    spearman_corr = spearman_corr if not np.isnan(spearman_corr) else 0.0
                    pearson_corr = pearson_corr if not np.isnan(pearson_corr) else 0.0
                    spearman_p = spearman_p if not np.isnan(spearman_p) else 1.0
                    pearson_p = pearson_p if not np.isnan(pearson_p) else 1.0
                    
                    correlations[rubric] = {
                        'spearman_r': spearman_corr,
                        'spearman_p': spearman_p,
                        'pearson_r': pearson_corr,
                        'pearson_p': pearson_p,
                        'mse': mean_squared_error(true_scores, pred_scores),
                        'mae': mean_absolute_error(true_scores, pred_scores),
                        'rmse': np.sqrt(mean_squared_error(true_scores, pred_scores)),
                        'r2': r2_score(true_scores, pred_scores),
                        'n_samples': len(pred_scores)
                    }
                    
                except Exception as e:
                    logger.error(f"Error calculating correlations for {rubric}: {e}")
                    correlations[rubric] = {
                        'spearman_r': 0.0,
                        'spearman_p': 1.0,
                        'pearson_r': 0.0,
                        'pearson_p': 1.0,
                        'mse': float('inf'),
                        'mae': float('inf'),
                        'rmse': float('inf'),
                        'r2': -float('inf'),
                        'n_samples': len(pred_scores)
                    }
        
        return correlations
    
    def get_ground_truth_scores(self, aggregation_method: str = 'mean') -> Dict[str, List[float]]:
        """
        Extract ground truth scores from the dataset.
        
        Args:
            aggregation_method: How to aggregate multiple annotator scores ('mean', 'median', 'max', 'min')
        
        Returns:
            Dict with rubric names as keys and aggregated scores as values
        """
        ground_truth = {rubric: [] for rubric in self.rubrics}
        
        for sample in self.dataset.data:
            rate_data = sample.get('rate', {})
            
            for rubric in self.rubrics:
                if rubric in rate_data:
                    scores = rate_data[rubric]
                    if isinstance(scores, list) and scores:
                        if aggregation_method == 'mean':
                            aggregated_score = np.mean(scores)
                        elif aggregation_method == 'median':
                            aggregated_score = np.median(scores)
                        elif aggregation_method == 'max':
                            aggregated_score = np.max(scores)
                        elif aggregation_method == 'min':
                            aggregated_score = np.min(scores)
                        else:
                            logger.warning(f"Unknown aggregation method: {aggregation_method}, using mean")
                            aggregated_score = np.mean(scores)
                        
                        ground_truth[rubric].append(aggregated_score)
                    else:
                        logger.warning(f"Invalid scores for {rubric} in sample {sample.get('id', 'unknown')}")
                        ground_truth[rubric].append(0.0)
                else:
                    # Rubric not present in this sample
                    ground_truth[rubric].append(0.0)
        
        return ground_truth
    
    def evaluate_model(self, model_predictions_file: str, aggregation_method: str = 'mean') -> Dict[str, Any]:
        """
        Evaluate a model's predictions from a JSON file.
        
        Args:
            model_predictions_file: Path to JSON file with predictions
            aggregation_method: How to aggregate ground truth scores
        
        Expected format: List of dicts with 'id' and rubric score predictions
        """
        try:
            with open(model_predictions_file, 'r', encoding='utf-8') as f:
                predictions_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading predictions: {e}")
            return {}
        
        # Organize predictions by rubric
        predictions = {rubric: [] for rubric in self.rubrics}
        ground_truth = {rubric: [] for rubric in self.rubrics}
        matched_samples = []
        
        for pred_sample in predictions_data:
            sample_id = pred_sample.get('id', '')
            gt_sample = self.dataset.get_sample_by_id(sample_id)
            
            if gt_sample is None:
                logger.warning(f"Sample {sample_id} not found in ground truth")
                continue
            
            matched_samples.append(sample_id)
            
            for rubric in self.rubrics:
                if rubric in pred_sample:
                    predictions[rubric].append(pred_sample[rubric])
                    
                    # Get ground truth score
                    gt_scores = gt_sample.get('rate', {}).get(rubric, [])
                    if isinstance(gt_scores, list) and gt_scores:
                        if aggregation_method == 'mean':
                            gt_score = np.mean(gt_scores)
                        elif aggregation_method == 'median':
                            gt_score = np.median(gt_scores)
                        elif aggregation_method == 'max':
                            gt_score = np.max(gt_scores)
                        elif aggregation_method == 'min':
                            gt_score = np.min(gt_scores)
                        else:
                            gt_score = np.mean(gt_scores)
                        
                        ground_truth[rubric].append(gt_score)
                    else:
                        ground_truth[rubric].append(0.0)
                else:
                    # Prediction not available for this rubric
                    logger.warning(f"No prediction for {rubric} in sample {sample_id}")
        
        # Calculate correlations
        correlations = self.calculate_correlations(predictions, ground_truth)
        
        # Add metadata
        evaluation_results = {
            'correlations': correlations,
            'metadata': {
                'total_predictions': len(predictions_data),
                'matched_samples': len(matched_samples),
                'match_rate': len(matched_samples) / len(predictions_data) if predictions_data else 0.0,
                'rubrics_evaluated': list(self.rubrics),
                'aggregation_method': aggregation_method
            }
        }
        
        return evaluation_results
    
    def evaluate_predictions_dict(self, predictions: Dict[str, Dict[str, float]], 
                                 aggregation_method: str = 'mean') -> Dict[str, Any]:
        """
        Evaluate predictions provided as a dictionary.
        
        Args:
            predictions: Dict with sample IDs as keys and rubric scores as values
            aggregation_method: How to aggregate ground truth scores
        
        Returns:
            Evaluation results dictionary
        """
        # Organize predictions by rubric
        pred_by_rubric = {rubric: [] for rubric in self.rubrics}
        gt_by_rubric = {rubric: [] for rubric in self.rubrics}
        matched_samples = []
        
        for sample_id, pred_scores in predictions.items():
            gt_sample = self.dataset.get_sample_by_id(sample_id)
            
            if gt_sample is None:
                logger.warning(f"Sample {sample_id} not found in ground truth")
                continue
            
            matched_samples.append(sample_id)
            
            for rubric in self.rubrics:
                if rubric in pred_scores:
                    pred_by_rubric[rubric].append(pred_scores[rubric])
                    
                    # Get ground truth score
                    gt_scores = gt_sample.get('rate', {}).get(rubric, [])
                    if isinstance(gt_scores, list) and gt_scores:
                        if aggregation_method == 'mean':
                            gt_score = np.mean(gt_scores)
                        elif aggregation_method == 'median':
                            gt_score = np.median(gt_scores)
                        elif aggregation_method == 'max':
                            gt_score = np.max(gt_scores)
                        elif aggregation_method == 'min':
                            gt_score = np.min(gt_scores)
                        else:
                            gt_score = np.mean(gt_scores)
                        
                        gt_by_rubric[rubric].append(gt_score)
                    else:
                        gt_by_rubric[rubric].append(0.0)
        
        # Calculate correlations
        correlations = self.calculate_correlations(pred_by_rubric, gt_by_rubric)
        
        # Add metadata
        evaluation_results = {
            'correlations': correlations,
            'metadata': {
                'total_predictions': len(predictions),
                'matched_samples': len(matched_samples),
                'match_rate': len(matched_samples) / len(predictions) if predictions else 0.0,
                'rubrics_evaluated': list(self.rubrics),
                'aggregation_method': aggregation_method
            }
        }
        
        return evaluation_results
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple model evaluation results.
        
        Args:
            model_results: Dict with model names as keys and evaluation results as values
        
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for model_name, results in model_results.items():
            correlations = results.get('correlations', {})
            metadata = results.get('metadata', {})
            
            for rubric, metrics in correlations.items():
                row = {
                    'model': model_name,
                    'rubric': rubric,
                    'spearman_r': metrics.get('spearman_r', 0.0),
                    'pearson_r': metrics.get('pearson_r', 0.0),
                    'mse': metrics.get('mse', float('inf')),
                    'mae': metrics.get('mae', float('inf')),
                    'rmse': metrics.get('rmse', float('inf')),
                    'r2': metrics.get('r2', -float('inf')),
                    'n_samples': metrics.get('n_samples', 0)
                }
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def plot_prediction_scatter(self, predictions: Dict[str, List[float]], 
                               ground_truth: Dict[str, List[float]], 
                               save_path: str = None) -> None:
        """Plot scatter plots of predictions vs ground truth for each rubric."""
        n_rubrics = len(self.rubrics)
        if n_rubrics == 0:
            logger.warning("No rubrics available for plotting")
            return
        
        n_cols = min(2, n_rubrics)
        n_rows = (n_rubrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_rubrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.ravel()
        
        for i, rubric in enumerate(self.rubrics):
            if i < len(axes) and rubric in predictions and rubric in ground_truth:
                pred_scores = predictions[rubric]
                true_scores = ground_truth[rubric]
                
                if len(pred_scores) > 0 and len(true_scores) > 0:
                    axes[i].scatter(true_scores, pred_scores, alpha=0.6)
                    axes[i].set_xlabel('Ground Truth')
                    axes[i].set_ylabel('Predictions')
                    axes[i].set_title(f'{rubric.replace("_", " ").title()}')
                    
                    # Add perfect prediction line
                    min_val = min(min(true_scores), min(pred_scores))
                    max_val = max(max(true_scores), max(pred_scores))
                    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
                    
                    # Add correlation coefficient
                    if len(pred_scores) >= 2:
                        try:
                            corr, _ = pearsonr(true_scores, pred_scores)
                            axes[i].text(0.05, 0.95, f'r = {corr:.3f}', 
                                       transform=axes[i].transAxes, 
                                       verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                        except:
                            pass
                    
                    axes[i].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(self.rubrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def export_evaluation_report(self, evaluation_results: Dict[str, Any], 
                                output_path: str, model_name: str = "Model") -> None:
        """Export evaluation results to a text file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"LecEval Model Evaluation Report: {model_name}\n")
            f.write("=" * 60 + "\n\n")
            
            # Metadata
            metadata = evaluation_results.get('metadata', {})
            f.write("EVALUATION METADATA\n")
            f.write("-" * 20 + "\n")
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Correlation results
            correlations = evaluation_results.get('correlations', {})
            f.write("CORRELATION RESULTS\n")
            f.write("-" * 20 + "\n")
            for rubric, metrics in correlations.items():
                f.write(f"\n{rubric.upper()}:\n")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"  {metric}: {value:.4f}\n")
                    else:
                        f.write(f"  {metric}: {value}\n")
            
            # Summary
            f.write("\nSUMMARY\n")
            f.write("-" * 10 + "\n")
            if correlations:
                avg_spearman = np.mean([m.get('spearman_r', 0) for m in correlations.values()])
                avg_pearson = np.mean([m.get('pearson_r', 0) for m in correlations.values()])
                avg_mae = np.mean([m.get('mae', 0) for m in correlations.values()])
                avg_rmse = np.mean([m.get('rmse', 0) for m in correlations.values()])
                
                f.write(f"Average Spearman Correlation: {avg_spearman:.4f}\n")
                f.write(f"Average Pearson Correlation: {avg_pearson:.4f}\n")
                f.write(f"Average MAE: {avg_mae:.4f}\n")
                f.write(f"Average RMSE: {avg_rmse:.4f}\n")
        
        logger.info(f"Evaluation report exported to {output_path}")