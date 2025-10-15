import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
from utils.data import TextDataset
from model.detector import Detector


class GPT2Detector(Detector):
    def __init__(self, config, tokenizer):
        # Build tokenizer and model with padding token included
        self.tokenizer = tokenizer

        self.model = GPT2LMHeadModel(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.train()

    def fit(self, X_train, y_train=None, epochs=10, batch_size=8, lr=5e-5, X_val=None):
        dataset = TextDataset(X_train, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = None
        if X_val is not None:
            val_dataset = TextDataset(X_val, self.tokenizer)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0.0
            self.model.train()
            for input_ids, attention_mask in dataloader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100  # Ignore padding in loss
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)

            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for input_ids, attention_mask in val_loader:
                        input_ids = input_ids.to(self.device)
                        attention_mask = attention_mask.to(self.device)
                        labels = input_ids.clone()
                        labels[attention_mask == 0] = -100  # Ignore padding in loss
                        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                        val_loss += outputs.loss.item()
                val_loss /= len(val_loader)
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - ValLoss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    def score(self, X):
        self.model.eval()
        scores = []
        for text in X:
            encoding = self.tokenizer(text, return_tensors='pt', truncation=True)
            input_ids = encoding.input_ids.to(self.device)
            labels = input_ids.clone()
            attention_mask = encoding.attention_mask.to(self.device)
            labels[attention_mask == 0] = -100  # Ignore padding in loss
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            perplexity = torch.exp(loss).item()
            scores.append(perplexity)
        return np.array(scores)
