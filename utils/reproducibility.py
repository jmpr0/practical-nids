import random
import numpy as np
import tensorflow as tf
import os


def set_seed(seed=42):
  """Set seed for reproducibility across Python, NumPy, and TensorFlow."""
  
  # Set Python's built-in random seed
  random.seed(seed)
  # Set NumPy seed
  np.random.seed(seed)
  # Set TensorFlow seed
  tf.random.set_seed(seed)

  # Ensure TensorFlow uses deterministic operations (for GPU computations)
  os.environ["PYTHONHASHSEED"] = str(seed)  # Fix hash-based randomness
  os.environ["TF_DETERMINISTIC_OPS"] = "1"  # Force deterministic operations
  os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # Use deterministic cuDNN algorithms
