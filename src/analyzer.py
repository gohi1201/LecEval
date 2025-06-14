"""
LecEval analysis module for statistical analysis and patterns
"""

from utils import *
from dataset import LecEvalDataset

class LecEvalAnalyzer:
    """Class for analyzing LecEval dataset statistics and patterns."""
    
    def __init__(self, dataset: LecEvalDataset):
        self.dataset = dataset
        self.df = self._create_dataframe()
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Create a pandas DataFrame from the dataset for easier analysis."""
        rows = []
        for sample in self.dataset.data:
            row = {
                'id': sample.get('id', ''),
                'transcript': sample.get('transcript', ''),
                'transcript_length': len(sample.get('transcript', '')),
                'slide_path': sample.get('slide', '')
            }
            
            # Add rubric scores (using mean of annotator ratings)
            rate_data = sample.get('rate', {})
            for rubric, scores in rate_data.items():
                if isinstance(scores, list) and scores:
                    row[f'{rubric}_mean'] = np.mean(scores)
                    row[f'{rubric}_std'] = np.std(scores) if len(scores) > 1 else 0.0
                    row[f'{rubric}_scores'] = scores
                    row[f'{rubric}_count'] = len(scores)
                else:
                    row[f'{rubric}_mean'] = 0.0
                    row[f'{rubric}_std'] = 0.0
                    row[f'{rubric}_scores'] = []
                    row[f'{rubric}_count'] = 0
            
            # Extract lecture and slide info from ID
            # Expected format: "ml-1_10_slide_000"
            id_str = sample.get('id', '')
            if '_slide_' in id_str:
                parts = id_str.split('_slide_')
                if len(parts) == 2:
                    row['lecture'] = parts[0]  # e.g., "ml-1_10"
                    row['slide_num'] = parts[1]  # e.g., "000"
            else:
                # Fallback parsing
                id_parts = id_str.split('_')
                if len(id_parts) >= 3:
                    row['lecture'] = '_'.join(id_parts[:-2])
                    row['slide_num'] = id_parts[-1]
                else:
                    row['lecture'] = id_str
                    row['slide_num'] = '0'
            
            # Extract course and section from lecture if possible
            if 'lecture' in row:
                lecture_parts = row['lecture'].split('_')
                if len(lecture_parts) >= 2:
                    row['course'] = lecture_parts[0]  # e.g., "ml-1"
                    row['section'] = lecture_parts[1] if len(lecture_parts) > 1 else '0'  # e.g., "10"
                else:
                    row['course'] = row['lecture']
                    row['section'] = '0'
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_available_rubrics(self) -> List[str]:
        """Get list of available rubrics from the dataset."""
        rubrics = set()
        for sample in self.dataset.data:
            rate_data = sample.get('rate', {})
            rubrics.update(rate_data.keys())
        return list(rubrics)
    
    def basic_statistics(self) -> Dict[str, Any]:
        """Generate basic dataset statistics."""
        available_rubrics = self.get_available_rubrics()
        
        stats = {
            'total_samples': len(self.df),
            'unique_lectures': self.df['lecture'].nunique() if 'lecture' in self.df.columns else 'N/A',
            'unique_courses': self.df['course'].nunique() if 'course' in self.df.columns else 'N/A',
            'transcript_length_stats': {
                'mean': self.df['transcript_length'].mean(),
                'std': self.df['transcript_length'].std(),
                'min': self.df['transcript_length'].min(),
                'max': self.df['transcript_length'].max(),
                'median': self.df['transcript_length'].median()
            },
            'rubric_statistics': {},
            'available_rubrics': available_rubrics
        }
        
        for rubric in available_rubrics:
            mean_col = f'{rubric}_mean'
            if mean_col in self.df.columns:
                stats['rubric_statistics'][rubric] = {
                    'mean': self.df[mean_col].mean(),
                    'std': self.df[mean_col].std(),
                    'min': self.df[mean_col].min(),
                    'max': self.df[mean_col].max(),
                    'median': self.df[mean_col].median(),
                    'count': self.df[f'{rubric}_count'].sum() if f'{rubric}_count' in self.df.columns else len(self.df)
                }
        
        return stats
    
    def plot_score_distributions(self, save_path: str = None) -> None:
        """Plot distribution of scores for each rubric."""
        available_rubrics = self.get_available_rubrics()
        n_rubrics = len(available_rubrics)
        
        if n_rubrics == 0:
            logger.warning("No rubrics found in dataset")
            return
        
        # Calculate subplot layout
        n_cols = min(2, n_rubrics)
        n_rows = (n_rubrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_rubrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.ravel()
        
        for i, rubric in enumerate(available_rubrics):
            mean_col = f'{rubric}_mean'
            if mean_col in self.df.columns and i < len(axes):
                data = self.df[mean_col].dropna()
                if len(data) > 0:
                    axes[i].hist(data, bins=min(20, len(data.unique())), alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'{rubric.replace("_", " ").title()} Distribution')
                    axes[i].set_xlabel('Score')
                    axes[i].set_ylabel('Frequency')
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add statistics text
                    mean_val = data.mean()
                    std_val = data.std()
                    axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
                    axes[i].legend()
        
        # Hide empty subplots
        for i in range(len(available_rubrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def correlation_analysis(self) -> pd.DataFrame:
        """Analyze correlations between different rubrics."""
        available_rubrics = self.get_available_rubrics()
        mean_cols = [f'{rubric}_mean' for rubric in available_rubrics 
                    if f'{rubric}_mean' in self.df.columns]
        
        if not mean_cols:
            logger.warning("No rubric mean columns found for correlation analysis")
            return pd.DataFrame()
        
        correlation_matrix = self.df[mean_cols].corr()
        
        # Rename columns for better readability
        new_names = {col: col.replace('_mean', '').replace('_', ' ').title() 
                    for col in mean_cols}
        correlation_matrix = correlation_matrix.rename(columns=new_names, index=new_names)
        
        return correlation_matrix
    
    def plot_correlation_heatmap(self, save_path: str = None) -> None:
        """Plot correlation heatmap between rubrics."""
        corr_matrix = self.correlation_analysis()
        
        if corr_matrix.empty:
            logger.warning("No correlation data available for plotting")
            return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, fmt='.3f')
        plt.title('Correlation Matrix of Evaluation Rubrics')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def lecture_wise_analysis(self) -> pd.DataFrame:
        """Analyze performance across different lectures."""
        if 'lecture' not in self.df.columns:
            logger.warning("Lecture information not available")
            return pd.DataFrame()
        
        available_rubrics = self.get_available_rubrics()
        lecture_stats = []
        
        for lecture in self.df['lecture'].unique():
            lecture_data = self.df[self.df['lecture'] == lecture]
            stats = {
                'lecture': lecture,
                'num_slides': len(lecture_data),
                'avg_transcript_length': lecture_data['transcript_length'].mean()
            }
            
            for rubric in available_rubrics:
                mean_col = f'{rubric}_mean'
                if mean_col in lecture_data.columns:
                    stats[f'{rubric}_avg'] = lecture_data[mean_col].mean()
                    stats[f'{rubric}_std'] = lecture_data[mean_col].std()
            
            lecture_stats.append(stats)
        
        return pd.DataFrame(lecture_stats).sort_values('lecture')
    
    def course_wise_analysis(self) -> pd.DataFrame:
        """Analyze performance across different courses."""
        if 'course' not in self.df.columns:
            logger.warning("Course information not available")
            return pd.DataFrame()
        
        available_rubrics = self.get_available_rubrics()
        course_stats = []
        
        for course in self.df['course'].unique():
            course_data = self.df[self.df['course'] == course]
            stats = {
                'course': course,
                'num_lectures': course_data['lecture'].nunique(),
                'num_slides': len(course_data),
                'avg_transcript_length': course_data['transcript_length'].mean()
            }
            
            for rubric in available_rubrics:
                mean_col = f'{rubric}_mean'
                if mean_col in course_data.columns:
                    stats[f'{rubric}_avg'] = course_data[mean_col].mean()
                    stats[f'{rubric}_std'] = course_data[mean_col].std()
            
            course_stats.append(stats)
        
        return pd.DataFrame(course_stats).sort_values('course')
    
    def get_samples_by_score_range(self, rubric: str, min_score: float = None, max_score: float = None) -> pd.DataFrame:
        """Get samples within a specific score range for a rubric."""
        mean_col = f'{rubric}_mean'
        if mean_col not in self.df.columns:
            logger.warning(f"Rubric '{rubric}' not found in dataset")
            return pd.DataFrame()
        
        filtered_df = self.df.copy()
        
        if min_score is not None:
            filtered_df = filtered_df[filtered_df[mean_col] >= min_score]
        if max_score is not None:
            filtered_df = filtered_df[filtered_df[mean_col] <= max_score]
        
        return filtered_df
    
    def annotator_agreement_analysis(self) -> Dict[str, Dict[str, float]]:
        """Analyze inter-annotator agreement for each rubric."""
        available_rubrics = self.get_available_rubrics()
        agreement_stats = {}
        
        for rubric in available_rubrics:
            scores_col = f'{rubric}_scores'
            std_col = f'{rubric}_std'
            
            if scores_col in self.df.columns:
                # Calculate agreement metrics
                std_values = self.df[std_col].dropna()
                agreement_stats[rubric] = {
                    'mean_std': std_values.mean(),
                    'median_std': std_values.median(),
                    'perfect_agreement_ratio': (std_values == 0).mean(),
                    'low_agreement_ratio': (std_values > 1.0).mean()
                }
        
        return agreement_stats
    
    def export_analysis_report(self, output_path: str) -> None:
        """Export comprehensive analysis report to a text file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("LecEval Dataset Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic statistics
            stats = self.basic_statistics()
            f.write("BASIC STATISTICS\n")
            f.write("-" * 20 + "\n")
            for key, value in stats.items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for subkey, subvalue in value.items():
                        f.write(f"  {subkey}: {subvalue}\n")
                else:
                    f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Correlation analysis
            corr_matrix = self.correlation_analysis()
            if not corr_matrix.empty:
                f.write("CORRELATION MATRIX\n")
                f.write("-" * 20 + "\n")
                f.write(corr_matrix.to_string())
                f.write("\n\n")
            
            # Lecture-wise analysis
            lecture_analysis = self.lecture_wise_analysis()
            if not lecture_analysis.empty:
                f.write("LECTURE-WISE ANALYSIS\n")
                f.write("-" * 20 + "\n")
                f.write(lecture_analysis.to_string(index=False))
                f.write("\n\n")
            
            # Annotator agreement
            agreement = self.annotator_agreement_analysis()
            if agreement:
                f.write("ANNOTATOR AGREEMENT ANALYSIS\n")
                f.write("-" * 30 + "\n")
                for rubric, metrics in agreement.items():
                    f.write(f"{rubric}:\n")
                    for metric, value in metrics.items():
                        f.write(f"  {metric}: {value:.4f}\n")
                f.write("\n")
        
        logger.info(f"Analysis report exported to {output_path}")