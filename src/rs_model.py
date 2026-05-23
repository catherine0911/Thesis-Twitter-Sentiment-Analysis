import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput

class RationaleSupervisedRoberta(nn.Module):
    def __init__(self, model_name, config, dropout_prob=0.1, rationale_dim=384): 
        """
        Rationale-Supervised RoBERTa model.
        Args:
            model_name: The HF model path/name.
            config: AutoConfig object.
            dropout_prob: Dropout probability for heads.
            rationale_dim: Output dimension for rationale embeddings (384 for ).
        """
        super(RationaleSupervisedRoberta, self).__init__()
        
        # Load the base RoBERTa model
        self.roberta = RobertaModel.from_pretrained(
            model_name, 
            config=config, 
            add_pooling_layer=False, 
            use_safetensors=True
        )
        self.dropout = nn.Dropout(dropout_prob)
        
        # Head 1: Sentiment Classification (3 classes: Neg, Neu, Pos)
        self.sentiment_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(config.hidden_size // 2, 3)
        )
        
        # Head 2: Rationale Projection (Projects hidden state to SentenceTransformer space)
        self.rationale_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(config.hidden_size // 2, rationale_dim)
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        """
        Forward pass.
        **kwargs is used to catch extra arguments like token_type_ids from tokenizer.
        """
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state 

        # Mean Pooling: Only pool over non-padding tokens
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        pooled_output = self.dropout(sum_embeddings / sum_mask)
        
        # 1. Generate Sentiment Logits
        sentiment_logits = self.sentiment_head(pooled_output)
        
        # 2. Generate Rationale Embeddings + L2 Normalization
        # Normalization is crucial for CosineEmbeddingLoss stability
        rat_output = self.rationale_head(pooled_output)
        rationale_embeddings = F.normalize(rat_output, p=2, dim=1)
        
        # Return sentiment wrapped in HF output class, and raw rationale tensor
        return SequenceClassifierOutput(logits=sentiment_logits), rationale_embeddings