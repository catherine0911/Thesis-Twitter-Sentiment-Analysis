import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput

class RationaleSupervisedRoberta(nn.Module):
    def __init__(self, model_name, config, dropout_prob=0.1, rationale_dim=384): 
        super(RationaleSupervisedRoberta, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained(
            model_name, config=config, add_pooling_layer=False, use_safetensors=True
        )
        self.dropout = nn.Dropout(dropout_prob)
        
        # Head 1: Rationale Projection (Remains the same)
        self.rationale_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(config.hidden_size // 2, rationale_dim)
        )

        # Sentiment Head takes (Hidden + Rationale) dimensions
        self.sentiment_head = nn.Sequential(
            nn.Linear(config.hidden_size + rationale_dim, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(config.hidden_size // 2, 3)
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state 

        # Mean Pooling
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        pooled_output = self.dropout(sum_embeddings / sum_mask)
        
        # Generate Rationale Embeddings first
        rat_output = self.rationale_head(pooled_output)
        rationale_embeddings = F.normalize(rat_output, p=2, dim=1)
        
        # Change based on Prototype test: Concatenate pooled [CLS] with the rationale embedding
        # This forces the sentiment head to condition on the rationale
        fused_output = torch.cat([pooled_output, rationale_embeddings], dim=-1)
        
        # 2. Generate Sentiment Logits from the fused representation
        sentiment_logits = self.sentiment_head(fused_output)
        
        return SequenceClassifierOutput(logits=sentiment_logits), rationale_embeddings