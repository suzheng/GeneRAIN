import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast
from einops import rearrange, repeat

from functools import partial
from contextlib import contextmanager

from local_attention import LocalAttention
from axial_positional_embedding import AxialPositionalEmbedding
import numpy as np

#from distutils.version import LooseVersion
from packaging.version import Version
#from torch.utils.checkpoint import checkpoint
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import BertConfig, BertModel, BertForMaskedLM
from transformers import GPT2Config, GPT2LMHeadModel
from dataclasses import dataclass, field
from typing import Optional, Tuple, Any, Dict
import torch.nn as nn
from torch import Tensor

def prepend_value_to_sequences(tensor, value, num_prepend):
    """
    Prepends a certain number of given values to the beginning of each sequence in a batch.
    """
    batch_size, seq_len = tensor.size()
    
    # Check the type of the tensor and prepare the corresponding prepend tensor
    if tensor.dtype == torch.bool:
        value = bool(value)
    elif tensor.dtype == torch.float32:
        value = float(value)
    elif tensor.dtype == torch.int64:
        value = int(value)
    elif tensor.dtype == torch.int32:
        value = int(value)
    else:
        raise ValueError(f"Unsupported tensor dtype: {tensor.dtype}")
    
    prepend_tensor = torch.full((batch_size, num_prepend), value, dtype=tensor.dtype).to(tensor.device)
    return torch.cat([prepend_tensor, tensor], dim=1)

def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val


@dataclass
class GeneExprTransformerConfig:
    num_tokens: int
    dim: int
    depth: int
    heads: int
    vocab_size: int
    n_positions: int
    dim_head: int = 64
    local_attn_heads: Tuple[int, ...] = (0,)
    local_window_size: int = 256
    causal: bool = False
    ff_mult: int = 4
    nb_features: Optional[int] = None
    feature_redraw_interval: int = 1000
    reversible: bool = False
    ff_chunks: int = 1
    ff_glu: bool = False
    emb_dropout: float = 0.
    ff_dropout: float = 0.
    attn_dropout: float = 0.
    generalized_attention: bool = False
    kernel_fn: nn.Module = nn.ReLU()
    use_scalenorm: bool = False
    use_rezero: bool = False
    cross_attend: bool = False
    no_projection: bool = False
    tie_embed: bool = False
    auto_check_redraw: bool = True
    qkv_bias: bool = False
    attn_out_bias: bool = False
    shift_tokens: bool = False
    pretrained_emb_path: Optional[str] = None
    pretrained_token_embedding_tensor: Optional[Tensor] = None
    expression_emb_type: str = "positional"
    number_of_bins_for_expression_embedding: int = 0
    bin_number_for_min_expr: int = 1
    gene_id_emb_requires_grad: bool = True
    expr_emb_requires_grad: bool = True
    do_embedding: bool = True
    transformer_model_name: Optional[str] = None
    layer_norm_eps: float = 1e-12
    norm_first: bool = True
    hidden_act: str = "gelu"
    extra_args: Dict[str, Any] = field(default_factory=dict)


# max_seq_len not needed anymore
class GeneExprTransformer(nn.Module):
    def __init__(self, config: GeneExprTransformerConfig):
        super().__init__()

        self.config = config
        self.local_attn_heads = cast_tuple(self.config.local_attn_heads)
        self.layer_norm_eps = self.config.layer_norm_eps
        self.transformer_model_name = self.config.transformer_model_name
        self.bin_number_for_min_expr = self.config.bin_number_for_min_expr
        self.number_of_bins_for_expression_embedding = self.config.number_of_bins_for_expression_embedding
        sliced_embeddings = None
        if self.config.pretrained_emb_path is not None:
            print(f"Reading in pretrained embedding from specified file at {self.config.pretrained_emb_path}!")
            pretrained_embeddings = torch.load(self.config.pretrained_emb_path)
            sliced_embeddings = pretrained_embeddings[:, :self.config.dim]

        if self.config.pretrained_token_embedding_tensor is not None:
            print(f"Use specified pretrained_token_embedding_tensor, set requires_grad to {self.config.gene_id_emb_requires_grad}!")
            self.token_emb = nn.Embedding.from_pretrained(self.config.pretrained_token_embedding_tensor)
            self.token_emb.weight.requires_grad = self.config.gene_id_emb_requires_grad
        elif self.config.pretrained_emb_path is not None and self.config.do_embedding:
            print(f"Use specified file at {self.config.pretrained_emb_path}, set requires_grad to {self.config.gene_id_emb_requires_grad}!")
            pretrained_embeddings = torch.load(self.config.pretrained_emb_path)
            sliced_embeddings = pretrained_embeddings[:, :self.config.dim]
            self.token_emb = nn.Embedding.from_pretrained(sliced_embeddings)
            self.token_emb.weight.requires_grad = self.config.gene_id_emb_requires_grad
        elif self.config.do_embedding:
            print(f"Use random token_emb, set requires_grad to {self.config.gene_id_emb_requires_grad}!")
            self.token_emb = nn.Embedding(self.config.num_tokens, self.config.dim)
            self.token_emb.weight.requires_grad = self.config.gene_id_emb_requires_grad

        if not(self.config.transformer_model_name in ["GPT", "Bert_pred_tokens", "BertExprInnerEmb"]):
            self.dropout = nn.Dropout(self.config.emb_dropout)
            self.norm = nn.LayerNorm(self.config.dim, eps=self.layer_norm_eps)

        if self.config.expression_emb_type == "positional" and self.config.do_embedding:
            pos_enc = self.positional_encoding(self.config.dim, self.config.number_of_bins_for_expression_embedding)
            self.expression_emb = nn.Embedding(self.config.number_of_bins_for_expression_embedding + 1, self.config.dim)
            self.expression_emb.weight.data.copy_(pos_enc)
            self.expression_emb.weight.requires_grad = self.config.expr_emb_requires_grad


        if self.config.transformer_model_name == "GPT":
            self.gpt = self.build_gpt2_lm_model(sliced_embeddings = sliced_embeddings)
        elif self.config.transformer_model_name in ["Bert_pred_tokens", "BertExprInnerEmb"]:
            self.bert_for_masked_lm_model = self.build_bert_for_masked_lm_model(sliced_embeddings = sliced_embeddings)


    def build_bert_for_masked_lm_model(self, sliced_embeddings=None):
        config = BertConfig(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.dim,
            num_hidden_layers=self.config.depth,
            num_attention_heads=self.config.heads,
            intermediate_size=self.config.dim * self.config.ff_mult,
            hidden_dropout_prob=self.config.ff_dropout,
            attention_probs_dropout_prob=self.config.attn_dropout,
            hidden_act=self.config.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            max_position_embeddings=self.config.n_positions
        )
        model = BertForMaskedLM(config)
        if self.config.pretrained_emb_path is not None:
            # Ensure that the dimensions of your pre-trained embeddings match the BERT model's embeddings
            if model.config.vocab_size == sliced_embeddings.size(0) and model.config.hidden_size == sliced_embeddings.size(1):
                print("Used pretrained emb for model.bert.embeddings.word_embeddings!")
                model.bert.embeddings.word_embeddings.weight = torch.nn.Parameter(sliced_embeddings)
            else:
                print("The dimensions of the pre-trained embeddings do not match the BERT model's dimensions!")
        return model   
    
    def build_gpt2_lm_model(self, sliced_embeddings=None):
        config = GPT2Config(
            vocab_size=self.config.vocab_size,
            n_positions=self.config.n_positions,
            n_embd=self.config.dim,
            n_layer=self.config.depth,
            n_head=self.config.heads,
            n_inner=self.config.dim * self.config.ff_mult,
            resid_pdrop=self.config.ff_dropout,
            attn_pdrop=self.config.attn_dropout,
            layer_norm_epsilon=self.layer_norm_eps,
            activation_function=self.config.hidden_act
        )
        model = GPT2LMHeadModel(config)
        if self.config.pretrained_emb_path is not None:
            # Ensure that the dimensions of your pre-trained embeddings match the GPT model's embeddings
            if model.config.vocab_size == sliced_embeddings.size(0) and model.config.n_embd == sliced_embeddings.size(1):
                print("Used pretrained emb for model.bert.embeddings.word_embeddings!")
                model.transformer.wte.weight = torch.nn.Parameter(sliced_embeddings)
            else:
                print("The dimensions of the pre-trained embeddings do not match the GPT model's dimensions!")
        return model

    #max_seq_len means number of bins for expression, should be 100 here but not 101
    @staticmethod
    def positional_encoding(dim, num_of_expr_bins, base=60.0):
        max_seq_len = num_of_expr_bins
        pe = np.zeros((max_seq_len + 1, dim))
        position = np.arange(0, max_seq_len, dtype=float)[:, np.newaxis]
        div_term = np.exp(np.arange(0, dim, 2) * -(np.log(base) / dim))
        pe[1:, 0::2] = np.sin(position * div_term)
        pe[1:, 1::2] = np.cos(position * div_term)

        # Add a unique encoding for masked expressions at index 0
        pe[0] = np.random.randn(dim)

        pe = torch.from_numpy(pe).float()
        return nn.Parameter(pe, requires_grad=True)


    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    # None of dataset will have special tokens, all special tokens will be added outside of GeneExprTransformer Module, inside the wrapper modules.
    # It supports 4 types of input:
    # 1. expression bin indices + gene ID indices
    # 2. expression and gene ID embedded (encoding_as_input == True)
    # 3. either of above + knockup/knockdown gene ID embedding. The forward function will add the expression bin embedding to get the perturb embedding, which will then be added to each of genes above.
    # 4. special token encoding, it will be prepend to the begining of sequence, should be done by the wrappers.
    def forward(self, x, expressions=None, return_type = "encoding", encoding_as_input = False, down_weighted_gene_emb_sum=None, up_weighted_gene_emb_sum=None, special_encoding_to_prepend=None, num_special_tokens=None, return_id_and_expr_embedded_only=False, **kwargs):
        device = x.device

        if self.transformer_model_name == "GPT":
            #print(kwargs['src_key_padding_mask'].dtype)
            outputs = self.gpt(x, **kwargs)
            return outputs
        #assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'
        if self.transformer_model_name in ["Bert_pred_tokens", "BertExprInnerEmb"]:
            #print(kwargs['src_key_padding_mask'].dtype)
            outputs = self.bert_for_masked_lm_model(x, **kwargs)
            return outputs
