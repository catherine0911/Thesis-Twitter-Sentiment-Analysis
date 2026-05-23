import torch
import torch.nn as nn
from transformers import RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput

class RobertaMTL(nn.Module):
    """ Multi-Task Learning (MTL) model with 2 heads: one for Sentiment and one for Sarcasm """
    def __init__(self, model_name, config, dropout_prob=0.1): 
        super(RobertaMTL, self).__init__()

        # Build shared Backbone - RoBERTa model
        self.roberta = RobertaModel.from_pretrained(
            model_name, 
            config=config, 
            add_pooling_layer=False, # Disable the default pooling layer to implement mean pooling
            use_safetensors=True
        )
        self.dropout = nn.Dropout(dropout_prob)
        # Sentiment head
        self.sentiment_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(config.hidden_size // 2, 3)
        )
        # Sarcasm head
        self.sarcasm_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(config.hidden_size // 2, 2)
        )

    def forward(self, input_ids, attention_mask, task='sentiment'):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state 
        # Mean Pooling
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        pooled_output = self.dropout(sum_embeddings / sum_mask)

        if task == 'sentiment':
            return SequenceClassifierOutput(logits=self.sentiment_head(pooled_output))
        else:
            return SequenceClassifierOutput(logits=self.sarcasm_head(pooled_output))