import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput

class CombinedRoberta(nn.Module):
    def __init__(self, model_name, config, dropout_prob=0.1, rationale_dim=384): 
        """
        Combined model with sentiment classification, sarcasm classification, and rationale embedding supervision.        
        """
        super(CombinedRoberta, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained(
            model_name, 
            config=config, 
            add_pooling_layer=False, 
            use_safetensors=True
        )
        self.dropout = nn.Dropout(dropout_prob)
        
        # Head 1: Sentiment Classification (3 classes)
        self.sentiment_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(config.hidden_size // 2, 3)
        )
        
        # Head 2: Sarcasm Classification (2 classes)
        self.sarcasm_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(config.hidden_size // 2, 2)
        )
        
        # Head 3: Shared Rationale Projection (Matches MiniLM 384-dim space)
        self.rationale_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(config.hidden_size // 2, rationale_dim)
        )

    def forward(self, input_ids, attention_mask, task='sentiment', **kwargs):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state 

        # Mean Pooling over non-padding tokens
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        pooled_output = self.dropout(sum_embeddings / sum_mask)
        
        # 1. Generate Rationale Embeddings (Shared across tasks)
        rat_output = self.rationale_head(pooled_output)
        rationale_embeddings = F.normalize(rat_output, p=2, dim=1)
        
        # 2. Route to the correct Classification Head
        if task == 'sentiment':
            logits = self.sentiment_head(pooled_output)
        elif task == 'sarcasm':
            logits = self.sarcasm_head(pooled_output)
        else:
            raise ValueError(f"Unknown task: {task}. Must be 'sentiment' or 'sarcasm'.")
            
        return SequenceClassifierOutput(logits=logits), rationale_embeddings