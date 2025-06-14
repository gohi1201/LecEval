"""
LecEval dataset handling module
"""

from utils import *

class LecEvalDataset:
    """Main class for handling LecEval dataset operations."""
    
    def __init__(self, data_path: str, images_path: str = None):
        """
        Initialize the LecEval dataset handler.
        
        Args:
            data_path: Path to the JSONL dataset file
            images_path: Path to the images directory (optional)
        """
        self.data_path = Path(data_path)
        self.images_path = Path(images_path) if images_path else None
        self.data = []
        self.rubrics = RUBRICS
        
        self.load_data()
    
    def load_data(self) -> None:
        """Load the dataset from JSONL file."""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            sample = json.loads(line)
                            self.data.append(sample)
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing JSON on line {line_num}: {e}")
                            continue
                            
            logger.info(f"Loaded {len(self.data)} samples from {self.data_path}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def get_sample(self, index: int) -> Dict[str, Any]:
        """Get a specific sample by index."""
        if 0 <= index < len(self.data):
            return self.data[index]
        else:
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.data)}")
    
    def get_sample_by_id(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """Get a sample by its ID."""
        for sample in self.data:
            if sample.get('id') == sample_id:
                return sample
        return None
    
    def get_image(self, sample: Dict[str, Any]) -> Optional[Image.Image]:
        """Load and return the slide image for a sample."""
        if not self.images_path:
            logger.warning("Images path not provided")
            return None
        
        # Handle slide path - remove leading slash if present
        slide_path = sample.get('slide', '')
        if slide_path.startswith('/'):
            slide_path = slide_path[1:]
        
        image_path = self.images_path / slide_path
        try:
            return Image.open(image_path)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def get_transcript(self, sample: Dict[str, Any]) -> str:
        """Extract transcript from sample."""
        return sample.get('transcript', '')
    
    def get_ratings(self, sample: Dict[str, Any]) -> Dict[str, List[int]]:
        """Extract ratings from sample."""
        return sample.get('rate', {})
    
    def get_prompt_conversation(self, sample: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract the prompt conversation from sample."""
        return sample.get('prompt', [])
    
    def get_evaluation_scores(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """Calculate mean scores for each evaluation criterion."""
        ratings = self.get_ratings(sample)
        scores = {}
        
        for criterion, values in ratings.items():
            if values:
                scores[criterion] = sum(values) / len(values)
            else:
                scores[criterion] = 0.0
                
        return scores
    
    def filter_by_score_range(self, criterion: str, min_score: float = None, max_score: float = None) -> List[Dict[str, Any]]:
        """Filter samples by score range for a specific criterion."""
        filtered_samples = []
        
        for sample in self.data:
            scores = self.get_evaluation_scores(sample)
            score = scores.get(criterion, 0.0)
            
            if min_score is not None and score < min_score:
                continue
            if max_score is not None and score > max_score:
                continue
                
            filtered_samples.append(sample)
        
        return filtered_samples
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics including score distributions."""
        stats = {
            'total_samples': len(self.data),
            'criteria_stats': {}
        }
        
        # Collect all scores for each criterion
        all_scores = {}
        for sample in self.data:
            scores = self.get_evaluation_scores(sample)
            for criterion, score in scores.items():
                if criterion not in all_scores:
                    all_scores[criterion] = []
                all_scores[criterion].append(score)
        
        # Calculate statistics for each criterion
        for criterion, scores in all_scores.items():
            if scores:
                stats['criteria_stats'][criterion] = {
                    'mean': sum(scores) / len(scores),
                    'min': min(scores),
                    'max': max(scores),
                    'count': len(scores)
                }
        
        return stats
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __iter__(self):
        """Make the dataset iterable."""
        return iter(self.data)