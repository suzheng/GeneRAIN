import torch.nn as nn
from models.GeneExprTransformer import GeneExprTransformer
from models.GeneExprTransformer import GeneExprTransformerConfig

class GeneExprTransformerBertExprInnerEmb(nn.Module):
    def __init__(self, config: GeneExprTransformerConfig):
        super(GeneExprTransformerBertExprInnerEmb, self).__init__()
        config.do_embedding = False
        # swap the token and position embeddings, as the Bert from the huggingface library predicts the tokens.
        config.max_position_embeddings = config.vocab_size
        config.n_positions = config.vocab_size
        config.vocab_size = config.number_of_bins_for_expression_embedding + 1
        self.gene_expr_transformer = GeneExprTransformer(config)
        #print(self.gene_expr_transformer)
                    
    def forward(self, x, expressions=None, labels=None,**kwargs):
        pos_ids = kwargs['position_ids']
        if labels != None:
            labels = labels.long()
        #print(f"forwarding in GeneExprTransformerBertPredTokens, x {x}\nlabels {labels} position_ids {pos_ids}")
        return self.gene_expr_transformer(x, labels=labels, **kwargs)
