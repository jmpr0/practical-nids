import numpy as np
from tensorflow.keras.models import Model
from model.detector import Detector
from model.autoencoder import AE


class KitNET(Detector):
  def __init__(self, input_size, k=5):
    """Creates a KitNET anomaly detection model with multiple autoencoders."""
    super().__init__()

    self.k = k

    # Determine input sizes for each ensemble autoencoder
    base_size = input_size // k
    self.input_sizes = [base_size + (1 if i < input_size % k else 0) for i in range(k)]

    # Precompute feature indices using NumPy
    self.splits = np.cumsum(self.input_sizes)[:-1]

    # Initialize autoencoder ensemble and output autoencoder
    self.ensemble = [AE(size) for size in self.input_sizes]
    self.output = AE(self.k)

  def compile(self, **kwargs):
    """Compile all autoencoders."""
    for ae in self.ensemble:
      ae.compile(**kwargs)
    self.output.compile(**kwargs)

  def fit(self, X, y, **kwargs):
    """Train all ensemble autoencoders and the final output autoencoder."""
    verbose = kwargs.get('verbose', False)
    X_splits = np.split(X, self.splits, axis=1)
    y_splits = np.split(y, self.splits, axis=1)
    y_score = np.zeros((X.shape[0], self.k))

    filenames_per_callback = {}

    for i, (X_sub, y_sub, ae) in enumerate(zip(X_splits, y_splits, self.ensemble)):
      if verbose:
        print(f'Training autoencoder {i + 1} / {self.k}...')

      # Modify callback filenames if applicable
      for callback in kwargs.get('callbacks', []):
        if hasattr(callback, 'filename') and isinstance(callback.filename, str):
          filename = filenames_per_callback.setdefault(callback.__class__.__name__, callback.filename)
          callback.filename = filename.replace('.csv', f'_{i}.csv')

      ae.fit(X_sub, y_sub, **kwargs)

      if verbose:
        print(f'Inferring autoencoder {i + 1} / {self.k}...')

      y_score[:, i] = ae.score(X_sub)

    if verbose:
      print('Training output autoencoder...')

    # Normalize and train the final output autoencoder
    y_score /= np.sum(y_score, axis=1, keepdims=True)
    self.output.fit(y_score, y_score, **kwargs)

  def score(self, X, **kwargs):
    """Compute anomaly scores using ensemble autoencoders and output autoencoder."""
    verbose = kwargs.get('verbose', False)
    X_splits = np.split(X, self.splits, axis=1)
    y_score = np.zeros((X.shape[0], self.k))

    for i, (X_sub, ae) in enumerate(zip(X_splits, self.ensemble)):
      if verbose:
        print(f'Inferring autoencoder {i + 1} / {self.k}...')

      y_score[:, i] = ae.score(X_sub)

    if verbose:
      print('Inferring output autoencoder...')

    y_score /= np.sum(y_score, axis=1, keepdims=True)
    return self.output.score(y_score, **kwargs)

  def print_summary(self):
    """Print summaries for all autoencoders."""
    for ae in self.ensemble:
      ae.print_summary()
    self.output.print_summary()
