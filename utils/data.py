import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from utils.logging import DiskLogger


# Data Loading
def load_data(dataset_name='kdd', num_packets=20, seed=0, validation_split=0.1, data_dir='dataset'):

  if dataset_name == 'kdd':

    train_fn = os.path.join(data_dir, 'NSL-KDD/KDDTrain+.txt')
    test_fn = os.path.join(data_dir, 'NSL-KDD/KDDTest+.txt')
    df_train = pd.read_csv(train_fn, header=None)
    df_test = pd.read_csv(test_fn, header=None)

    X_train, y_train = df_train.iloc[:, :-2].to_numpy(), df_train.iloc[:, -2].to_numpy()
    X_test, y_test = df_test.iloc[:, :-2].to_numpy(), df_test.iloc[:, -2].to_numpy()

  elif dataset_name == 'iot-23':

    base_path = os.path.join(data_dir, 'IoT-23')
    path = os.path.join(base_path, f'iot23-stats_{num_packets}-pkts.parquet')

    df = pd.read_parquet(path)
    X, y = df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)

  X_val, y_val = None, None
  if validation_split:
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_split, stratify=y_train, random_state=seed)

  return X_train, y_train, X_test, y_test, X_val, y_val

# Data Preprocessing
def preprocess_data(X_train, y_train, X_test, y_test, X_val=None, y_val=None, dataset_name='kdd'):

  if dataset_name == 'kdd':

    cat_indexes = [i for i, v in enumerate(X_train[0]) if isinstance(v, str)]
    for cat_index in cat_indexes:
      oe = OrdinalEncoder()
      X_train[:, cat_index] = oe.fit_transform(X_train[:, cat_index].reshape(-1, 1)).squeeze()
      if X_val is not None:
        X_val[:, cat_index] = oe.transform(X_val[:, cat_index].reshape(-1, 1)).squeeze()
      X_test[:, cat_index] = oe.transform(X_test[:, cat_index].reshape(-1, 1)).squeeze()
    legitimate_class = 'normal'

  else:  # dataset_name == 'iot-23'

    legitimate_class = 'Benign'

  # Normalize features
  scaler = MinMaxScaler(feature_range=(0, 1))
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)
  if X_val is not None:
    X_val = scaler.transform(X_val)

  # Encode labels
  label_encoder = LabelEncoder().fit(np.concatenate((y_train, y_test)))
  y_train = label_encoder.transform(y_train)
  y_test = label_encoder.transform(y_test)
  if y_val is not None:
    y_val = label_encoder.transform(y_val)

  encoded_legit_class = label_encoder.transform([legitimate_class])[0]

  return X_train, y_train, X_test, y_test, X_val, y_val, encoded_legit_class

def log(results_path, config):
    
    # Create the results directory if it does not exist
    os.makedirs(results_path, exist_ok=True)

    # Initialize the disk logger to save experiment details
    logger = DiskLogger(results_path)

    logger.log_args(config)

def build_vocab(texts):
    vocab = set()
    for text in texts:
        tokens = text.split()
        vocab.update(tokens)
    token2id = {token: idx for idx, token in enumerate(sorted(vocab))}
    return token2id, len(vocab)
    
def tabular_to_text(X):
    # Convert numpy array row to string with space-separated numeric values
    texts = []
    for row in X:
        text = " ".join([str(round(x, 4)) for x in row])
        texts.append(text)
    return texts

class SimpleTokenizer:
    def __init__(self, token2id):
        self.token2id = dict(token2id)  # Ensure we have a copy to avoid modifying the original
        self.pad_token = '<PAD>'
        if self.pad_token not in self.token2id:
            self.pad_token_id = len(self.token2id)
            self.token2id[self.pad_token] = self.pad_token_id
        else:
            self.pad_token_id = self.token2id[self.pad_token]
        self.id2token = {v: k for k, v in self.token2id.items()}

    def encode(self, text, max_length=128, truncation=True, padding="max_length", return_tensors=None):
        tokens = text.split()
        token_ids = [self.token2id.get(t, self.pad_token_id) for t in tokens]

        if truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        pad_len = max_length - len(token_ids)
        if padding == "max_length":
            token_ids += [self.pad_token_id] * pad_len

        attention_mask = [1] * (max_length - pad_len) + [0] * pad_len

        import torch
        input_ids_tensor = torch.tensor([token_ids])
        attention_mask_tensor = torch.tensor([attention_mask])

        class Encoding:
            pass

        encoding = Encoding()
        encoding.input_ids = input_ids_tensor
        encoding.attention_mask = attention_mask_tensor
        return encoding

    def __call__(self, text, max_length=128, truncation=True, padding="max_length", return_tensors=None):
        return self.encode(text, max_length, truncation, padding, return_tensors)
    
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(self.texts[idx], max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        input_ids = encoded.input_ids.squeeze(0)
        attention_mask = encoded.attention_mask.squeeze(0)
        return input_ids, attention_mask
