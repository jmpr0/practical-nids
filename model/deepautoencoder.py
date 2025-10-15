import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from model.detector import Detector


class DAE(Detector):
  def __init__(self, input_size, red_fact=[0.75, 0.5], encoder_activation='relu', decoder_activation='relu', final_activation='sigmoid'):
    """Creates a Deep Autoencoder inside the Detector class.

    Parameters:
    - input_size (int): Number of input features.
    - red_fact (list of float): List of reduction factors for encoder layers.
    - encoder_activation (str): Activation function for encoder layers (default: 'relu').
    - decoder_activation (str): Activation function for decoder layers (default: 'relu').
    - final_activation (str): Activation function for the final decoder layer (default: 'sigmoid').
    """
    super().__init__()

    # Compute encoder and decoder sizes
    encoder_sizes = [int(np.ceil(input_size * rf)) for rf in red_fact]
    decoder_sizes = encoder_sizes[::-1][1:]  # Exclude the last encoder size

    # Define encoder layers
    inputs = Input(shape=(input_size,))
    x = inputs
    for size in encoder_sizes:
      x = Dense(size, activation=encoder_activation)(x)

    # Define decoder layers
    for size in decoder_sizes:
      x = Dense(size, activation=decoder_activation)(x)

    # Final output layer
    x = Dropout(0.2)(x)
    outputs = Dense(input_size, activation=final_activation)(x)

    self.model = Model(inputs, outputs)
