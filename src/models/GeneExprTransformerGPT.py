import torch.nn as nn
from models.GeneExprTransformer import GeneExprTransformer
from models.GeneExprTransformer import GeneExprTransformerConfig

class GeneExprTransformerGPT(nn.Module):
    def __init__(self, config: GeneExprTransformerConfig):
        super(GeneExprTransformerGPT, self).__init__()
        config.do_embedding = False
        self.gene_expr_transformer = GeneExprTransformer(config)
                    
    def forward(self, x, expressions=None, labels=None, **kwargs):
        return self.gene_expr_transformer(x, labels=x.long(), **kwargs)
