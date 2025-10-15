import numpy as np


class Detector:
  """A general anomaly detector using TensorFlow/Keras models."""

  def __init__(self):
    self.model = None

  def compile(self, **kwargs):
    """Compile the Keras model with the given optimizer and loss function."""
    if self.model:
      self.model.compile(**kwargs)

  def fit(self, X, y, **kwargs):
    """Train the model with input features X and target y."""
    if self.model:
      self.model.fit(X, y, **kwargs)

  def predict(self, X, **kwargs):
    """Generate predictions from the trained model."""
    return self.model.predict(X, **kwargs) if self.model else None

  def score(self, X, **kwargs):
    """Calculate the root mean squared error (RMSE) as an anomaly score."""
    X_pred = self.predict(X, **kwargs)
    if X_pred is not None:
      return np.sqrt(np.mean((X - X_pred) ** 2, axis=1))  # Vectorized RMSE
    return None

  def print_summary(self):
    """Prints the model architecture summary."""
    if self.model:
      self.model.summary()
