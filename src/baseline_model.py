import torch
import torch.nn as nn
from transformers import RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput

class RobertaBaseline(nn.Module):
    """
    Single-task RoBERTa sentiment classifier used as the baseline model.
    """
    def __init__(self, model_name, config, dropout_prob=0.1):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(
            model_name, config=config,
            add_pooling_layer=False, use_safetensors=True
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(config.hidden_size // 2, 3)
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        last_h = outputs.last_hidden_state
        
        # Mean-pool token embeddings while ignoring padding tokens
        mask = attention_mask.unsqueeze(-1).expand(last_h.size()).float()
        pooled = torch.sum(last_h * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        
        logits = self.classifier(self.dropout(pooled))
        return SequenceClassifierOutput(logits=logits)