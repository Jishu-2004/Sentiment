# Stocks_M.py
import torch
import torch.nn as nn
from transformers import DistilBertModel

class StocksModel(nn.Module):
    def __init__(self, num_labels=3):
        super(StocksModel, self).__init__()

        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # Stock market sentiment = jargon + noise → medium dropout
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(self.encoder.config.dim, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_token = outputs.last_hidden_state[:, 0]
        cls_token = self.dropout(cls_token)

        logits = self.classifier(cls_token)
        return logits
