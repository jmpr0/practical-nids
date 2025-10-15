import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from model.detector import Detector


class AE(Detector):
  def __init__(self, input_size, red_fact=0.5, encoder_activation='relu', decoder_activation='sigmoid'):
    """Creates an Autoencoder model inside the Detector class.

    Parameters:
    - input_size (int): Number of input features.
    - red_fact (float): Reduction factor for encoding layer size (default: 0.5).
    - encoder_activation (str): Activation function for encoder layer (default: 'relu').
    - decoder_activation (str): Activation function for decoder layer (default: 'sigmoid').
    """
    super().__init__()

    reduced_size = int(np.ceil(input_size * red_fact))

    # Define autoencoder architecture
    inputs = Input(shape=(input_size,))
    encoded = Dense(reduced_size, activation=encoder_activation)(inputs)
    encoded = Dropout(0.2)(encoded)
    decoded = Dense(input_size, activation=decoder_activation)(encoded)

    # Assign model to self.model
    self.model = Model(inputs, decoded)
