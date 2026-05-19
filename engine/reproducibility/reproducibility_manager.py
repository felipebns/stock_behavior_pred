import os
import random
import numpy as np


class ReproducibilityManager:
    """Centralizes all randomness control for deterministic results."""
    
    @classmethod
    def setup_reproducibility(cls, seed: int = 42) -> None:
        """
        Setup reproducibility by setting all random seeds.
        
        CRITICAL: Must be called BEFORE importing ML libraries.
        """
        # Python random module
        random.seed(seed)
        
        # NumPy random
        np.random.seed(seed)
        
        # Python hash seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Joblib (scikit-learn parallelization)
        os.environ['JOBLIB_RANDOM_SEED'] = str(seed)