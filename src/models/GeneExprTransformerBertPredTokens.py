import torch.nn as nn
from models.GeneExprTransformer import GeneExprTransformer
from models.GeneExprTransformer import GeneExprTransformerConfig

class GeneExprTransformerBertPredTokens(nn.Module):
    def __init__(self, config: GeneExprTransformerConfig):
        super(GeneExprTransformerBertPredTokens, self).__init__()
        config.do_embedding = False
        self.gene_expr_transformer = GeneExprTransformer(config)
                    
    def forward(self, x, expressions=None, labels=None,**kwargs):
        if labels != None:
            labels = labels.long()
        return self.gene_expr_transformer(x, labels=labels, **kwargs)
