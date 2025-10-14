from time import process_time
from tensorflow.keras.callbacks import Callback


class TimeEpochs(Callback):
  """Callback to measure per-epoch training time."""

  def __init__(self):
    super().__init__()
    self.epoch_time_start = None

  def on_epoch_begin(self, epoch, logs=None):
    self.epoch_time_start = process_time()

  def on_epoch_end(self, epoch, logs=None):
    if logs is not None:
      logs['time'] = process_time() - self.epoch_time_start

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

