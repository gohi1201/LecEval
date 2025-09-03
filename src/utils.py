"""
Common utilities and configurations for LecEval
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

def setup_logging(log_file: str = None, level: int = logging.INFO):
    """
    Configure logging for LecEval.
    If log_file is provided, logs will be written to that file as well.
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Common constants
RUBRICS = ['content_relevance', 'expressive_clarity', 'logical_structure', 'audience_engagement'] 