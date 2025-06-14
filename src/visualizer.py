"""
LecEval visualization module for creating visualizations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image
import os
import logging

logger = logging.getLogger(__name__)

class LecEvalVisualizer:
    """Class for creating visualizations of the LecEval dataset."""
    
    def __init__(self, dataset, analyzer=None):
        """
        Initialize the visualizer.
        
        Args:
            dataset: LecEvalDataset instance
            analyzer: LecEvalAnalyzer instance (optional)
        """
        self.dataset = dataset
        self.analyzer = analyzer
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def create_sample_visualization(self, sample_id: str, save_path: str = None) -> None:
        """
        Create a comprehensive visualization for a specific sample.
        
        Args:
            sample_id: ID of the sample to visualize
            save_path: Optional path to save the visualization
        """
        sample = self.dataset.get_sample_by_id(sample_id)
        if not sample:
            logger.error(f"Sample {sample_id} not found")
            return
        
        # Create figure with proper subplot layout
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # 1. Display slide image (top left)
        ax_img = fig.add_subplot(gs[0, 0])
        self._plot_slide_image(ax_img, sample)
        
        # 2. Display transcript text (top right)
        ax_text = fig.add_subplot(gs[0, 1])
        self._plot_transcript_text(ax_text, sample)
        
        # 3. Plot rubric scores with error bars (middle left)
        ax_scores = fig.add_subplot(gs[1, 0])
        self._plot_rubric_scores(ax_scores, sample)
        
        # 4. Show individual annotator scores heatmap (middle right)
        ax_heatmap = fig.add_subplot(gs[1, 1])
        self._plot_annotator_heatmap(ax_heatmap, sample)
        
        # 5. Show score distribution (bottom spanning both columns)
        ax_dist = fig.add_subplot(gs[2, :])
        self._plot_score_distribution(ax_dist, sample)
        
        # Add main title
        fig.suptitle(f'Sample Analysis: {sample_id}', fontsize=18, fontweight='bold')
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def _plot_slide_image(self, ax, sample):
        """Plot the slide image if available."""
        try:
            if hasattr(self.dataset, 'images_path') and self.dataset.images_path:
                # Construct image path from sample data
                if 'slide' in sample:
                    img_path = os.path.join(self.dataset.images_path, sample['slide'].lstrip('/'))
                    if os.path.exists(img_path):
                        img = Image.open(img_path)
                        ax.imshow(img)
                        ax.set_title('Lecture Slide', fontsize=14, fontweight='bold')
                        ax.axis('off')
                        return
            
            # If no image available, show placeholder
            ax.text(0.5, 0.5, 'Slide Image\nNot Available', 
                   ha='center', va='center', fontsize=14,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title('Lecture Slide', fontsize=14, fontweight='bold')
            ax.axis('off')
            
        except Exception as e:
            logger.warning(f"Error loading slide image: {e}")
            ax.text(0.5, 0.5, 'Error Loading\nSlide Image', 
                   ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
    
    def _plot_transcript_text(self, ax, sample):
        """Plot the transcript text."""
        transcript = sample.get('transcript', 'No transcript available')
        
        # Wrap text if it's too long
        if len(transcript) > 500:
            transcript = transcript[:500] + "..."
        
        ax.text(0.05, 0.95, transcript, 
               transform=ax.transAxes, 
               verticalalignment='top', 
               horizontalalignment='left',
               wrap=True, 
               fontsize=10,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        ax.set_title('Speech Transcript', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def _plot_rubric_scores(self, ax, sample):
        """Plot rubric scores with error bars."""
        if 'rate' not in sample:
            ax.text(0.5, 0.5, 'No rating data available', ha='center', va='center')
            ax.set_title('Rubric Scores', fontsize=14, fontweight='bold')
            return
        
        rubrics = list(sample['rate'].keys())
        scores = [np.mean(sample['rate'][rubric]) for rubric in rubrics]
        stds = [np.std(sample['rate'][rubric]) if len(sample['rate'][rubric]) > 1 else 0 
                for rubric in rubrics]
        
        # Create bar plot with error bars
        bars = ax.bar(range(len(rubrics)), scores, yerr=stds, 
                     capsize=5, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Color bars based on score
        for bar, score in zip(bars, scores):
            if score >= 4:
                bar.set_color('green')
            elif score >= 3:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax.set_xticks(range(len(rubrics)))
        ax.set_xticklabels([r.replace('_', '\n').title() for r in rubrics], 
                          rotation=0, fontsize=10)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Rubric Scores (Mean ± Std)', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 5.5)
        ax.grid(axis='y', alpha=0.3)
        
        # Add score labels on bars
        for i, (score, std) in enumerate(zip(scores, stds)):
            ax.text(i, score + std + 0.1, f'{score:.1f}', 
                   ha='center', va='bottom', fontweight='bold')
    
    def _plot_annotator_heatmap(self, ax, sample):
        """Plot individual annotator scores as a heatmap."""
        if 'rate' not in sample:
            ax.text(0.5, 0.5, 'No rating data available', ha='center', va='center')
            ax.set_title('Individual Annotator Scores', fontsize=14, fontweight='bold')
            return
        
        # Prepare data for heatmap
        rubrics = list(sample['rate'].keys())
        max_annotators = max(len(sample['rate'][rubric]) for rubric in rubrics)
        
        # Create matrix for heatmap
        data_matrix = []
        labels = []
        
        for rubric in rubrics:
            scores = sample['rate'][rubric]
            # Pad with NaN if fewer annotators for this rubric
            padded_scores = scores + [np.nan] * (max_annotators - len(scores))
            data_matrix.append(padded_scores)
            labels.append(rubric.replace('_', ' ').title())
        
        data_matrix = np.array(data_matrix)
        
        # Create heatmap
        sns.heatmap(data_matrix, 
                   annot=True, 
                   fmt='.0f',
                   cmap='RdYlGn', 
                   ax=ax, 
                   cbar_kws={'label': 'Score'},
                   yticklabels=labels,
                   xticklabels=[f'Annotator {i+1}' for i in range(max_annotators)],
                   vmin=1, vmax=5)
        
        ax.set_title('Individual Annotator Scores', fontsize=14, fontweight='bold')
        ax.set_xlabel('Annotators', fontsize=12)
        ax.set_ylabel('Rubric Criteria', fontsize=12)
    
    def _plot_score_distribution(self, ax, sample):
        """Plot the distribution of all scores."""
        if 'rate' not in sample:
            ax.text(0.5, 0.5, 'No rating data available', ha='center', va='center')
            ax.set_title('Score Distribution', fontsize=14, fontweight='bold')
            return
        
        # Collect all scores
        all_scores = []
        rubric_labels = []
        
        for rubric, scores in sample['rate'].items():
            all_scores.extend(scores)
            rubric_labels.extend([rubric.replace('_', ' ').title()] * len(scores))
        
        if not all_scores:
            ax.text(0.5, 0.5, 'No scores to display', ha='center', va='center')
            return
        
        # Create violin plot
        df = pd.DataFrame({'Score': all_scores, 'Rubric': rubric_labels})
        sns.violinplot(data=df, x='Rubric', y='Score', ax=ax, inner='points')
        
        ax.set_title('Score Distribution by Rubric', fontsize=14, fontweight='bold')
        ax.set_xlabel('Rubric Criteria', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0.5, 5.5)
        ax.grid(axis='y', alpha=0.3)
        
        # Rotate x-axis labels if needed
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def create_dataset_overview(self, save_path: str = None) -> None:
        """
        Create an overview visualization of the entire dataset.
        
        Args:
            save_path: Optional path to save the visualization
        """
        if not hasattr(self.dataset, 'data') or not self.dataset.data:
            logger.error("No data available for overview")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('LecEval Dataset Overview', fontsize=16, fontweight='bold')
        
        # 1. Score distribution across all samples
        self._plot_overall_score_distribution(axes[0, 0])
        
        # 2. Average scores by rubric
        self._plot_average_scores_by_rubric(axes[0, 1])
        
        # 3. Inter-annotator agreement
        self._plot_inter_annotator_agreement(axes[1, 0])
        
        # 4. Sample count statistics
        self._plot_sample_statistics(axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Overview visualization saved to {save_path}")
        
        plt.show()
    
    def _plot_overall_score_distribution(self, ax):
        """Plot distribution of all scores in the dataset."""
        all_scores = []
        for sample in self.dataset.data:
            if 'rate' in sample:
                for scores in sample['rate'].values():
                    all_scores.extend(scores)
        
        if all_scores:
            ax.hist(all_scores, bins=5, range=(0.5, 5.5), alpha=0.7, edgecolor='black')
            ax.set_xlabel('Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Overall Score Distribution')
            ax.set_xticks(range(1, 6))
        else:
            ax.text(0.5, 0.5, 'No scores available', ha='center', va='center')
    
    def _plot_average_scores_by_rubric(self, ax):
        """Plot average scores for each rubric across all samples."""
        rubric_scores = {}
        
        for sample in self.dataset.data:
            if 'rate' in sample:
                for rubric, scores in sample['rate'].items():
                    if rubric not in rubric_scores:
                        rubric_scores[rubric] = []
                    rubric_scores[rubric].extend(scores)
        
        if rubric_scores:
            rubrics = list(rubric_scores.keys())
            avg_scores = [np.mean(rubric_scores[rubric]) for rubric in rubrics]
            
            bars = ax.bar(range(len(rubrics)), avg_scores, alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(rubrics)))
            ax.set_xticklabels([r.replace('_', '\n').title() for r in rubrics])
            ax.set_ylabel('Average Score')
            ax.set_title('Average Scores by Rubric')
            ax.set_ylim(0, 5)
            
            # Add value labels on bars
            for bar, score in zip(bars, avg_scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{score:.2f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No rubric data available', ha='center', va='center')
    
    def _plot_inter_annotator_agreement(self, ax):
        """Plot inter-annotator agreement statistics."""
        agreements = []
        
        for sample in self.dataset.data:
            if 'rate' in sample:
                for scores in sample['rate'].values():
                    if len(scores) > 1:
                        agreements.append(np.std(scores))
        
        if agreements:
            ax.hist(agreements, bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Standard Deviation of Scores')
            ax.set_ylabel('Frequency')
            ax.set_title('Inter-Annotator Agreement\n(Lower std = Higher agreement)')
            ax.axvline(np.mean(agreements), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(agreements):.2f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No multi-annotator data', ha='center', va='center')
    
    def _plot_sample_statistics(self, ax):
        """Plot basic dataset statistics."""
        stats = {
            'Total Samples': len(self.dataset.data),
            'Samples with Ratings': sum(1 for s in self.dataset.data if 'rate' in s),
            'Samples with Transcripts': sum(1 for s in self.dataset.data if 'transcript' in s),
            'Samples with Slides': sum(1 for s in self.dataset.data if 'slide' in s)
        }
        
        ax.bar(range(len(stats)), list(stats.values()), alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(stats)))
        ax.set_xticklabels(list(stats.keys()), rotation=45, ha='right')
        ax.set_ylabel('Count')
        ax.set_title('Dataset Statistics')
        
        # Add value labels on bars
        for i, value in enumerate(stats.values()):
            ax.text(i, value + 0.5, str(value), ha='center', va='bottom')