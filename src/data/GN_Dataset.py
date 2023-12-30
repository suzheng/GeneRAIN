from utils.params import params
params = params()
import h5py
import torch
import numpy as np
import pandas as pd
import sys
from torch.utils.data import Dataset
from data.BertMasking import get_bert_masking
from data.discretize_expression import discretize_expression_zscores, discretize_expression_uniform
from data.discretize_expression import uniform_bin_count_keep_ones
from data.data_utils import get_top_genes
from utils.json_utils import JsonUtils
from train.common_params_funs import config, get_gene2idx
from train.common import train

from utils.config_loader import Config
ju = JsonUtils()
config = Config()
proj_path = config.proj_path
gene_to_idx, _ = get_gene2idx()

# None of dataset will have special tokens, all special tokens will be added outside of GeneExprTransformer Module, inside the wrapper modules.
class GN_Dataset(Dataset):
    def __init__(self, 
    sample_by_gene_expr_mat=None,
    gene_symbols=None,
    n_bins=params.NUM_BINS, 
    mask_fraction=params.MASK_FRACTIONS[0], 
    expr_discretization_method=params.EXPR_DISCRETIZATION_METHOD, 
    num_of_genes=params.NUM_OF_GENES_SELECTED,
    number_of_special_embeddings=params.NUMBER_OF_SPECIAL_TOKEN_IN_DATASET,
    sort_return_expr_numerically=(params.TRANSFORMER_MODEL_NAME == "GPT" or params.TRANSFORMER_MODEL_NAME == "Bert_pred_tokens")
    ):
        # self.gene_symbols = np.array([gene_symbol.decode('utf-8') for gene_symbol in gene_symbols])
        assert len(gene_symbols) == sample_by_gene_expr_mat.shape[1], "len(gene_symbols) != sample_by_gene_expr_mat.shape[1]!"
        self.gene_symbols = np.array(gene_symbols)
        sample_by_gene_expr_mat = np.array(sample_by_gene_expr_mat)
        del gene_symbols
        self.n_bins = n_bins
        self.num_of_genes_to_return = num_of_genes
        self.sort_return_expr_numerically = sort_return_expr_numerically
        # Find the index of genes in gene_to_idx
        self.gene_symbol_indices = np.array([gene_to_idx.get(gene_symbol, -1) for gene_symbol in self.gene_symbols], dtype=np.int32)
        self.mask_fraction = mask_fraction
        # Filter gene_symbol_indices to keep only the genes with index in gene_to_idx
        self.genes_in_gene_emb = self.gene_symbol_indices != -1
        self.number_of_special_embeddings = number_of_special_embeddings
        self.bert_masking = get_bert_masking(mask_fraction=self.mask_fraction)
        self.gene_indices_for_ret = torch.tensor(self.gene_symbol_indices[self.genes_in_gene_emb], dtype=torch.int32)
        self.expr_discretization_method = expr_discretization_method
        self.gene_symbols_in_dataset = [gene for gene in self.gene_symbols[self.genes_in_gene_emb]]
        self.filtered_expression_data = sample_by_gene_expr_mat[:, self.genes_in_gene_emb]
        del sample_by_gene_expr_mat
        self.sample_size = self.filtered_expression_data.shape[0]
        
    def __len__(self):
        return self.sample_size

    def prepend_special_tokens(self, result, number_of_special_embeddings):
        if number_of_special_embeddings == 0:
            return result
        # Prepend indices for gene_indices
        special_tokens_indices = torch.arange(number_of_special_embeddings, dtype=result['gene_indices'].dtype)
        result['gene_indices'] = torch.cat([special_tokens_indices, result['gene_indices']])

        # Prepend zeros for input_binned_expr and output_binned_expr
        zeros_to_prepend = torch.zeros(number_of_special_embeddings, dtype=result['masked_expression'].dtype)
        result['masked_expression'] = torch.cat([zeros_to_prepend, result['masked_expression']])
        result['true_expression'] = torch.cat([zeros_to_prepend, result['true_expression']])
        
        # Prepend False for zero_expression_genes
        false_to_prepend = torch.zeros(number_of_special_embeddings, dtype=torch.bool)
        result['zero_expression_genes'] = torch.cat([false_to_prepend, result['zero_expression_genes']])
        
        return result
    
    
    def __getitem__(self, idx):
        return self.__getitem_only_some_genes__(idx)
        
    def sort_result_based_on_expr(self, result, expression_vector_float32):
        if not torch.is_tensor(expression_vector_float32):
            expression_vector_float32 = torch.tensor(expression_vector_float32)
        sorted_indices = torch.argsort(expression_vector_float32, descending=True)
        for key, value in result.items():
            if value.numel() == expression_vector_float32.numel():
                result[key] = value[sorted_indices].clone().detach()
        return result

    def __getitem_only_some_genes__(self, idx):

        expression_vector = self.filtered_expression_data[idx, :]

        expression_vector_float32 = expression_vector.astype(np.float32)

        expression_vector_float32, top_abs_indices = get_top_genes(expression_vector_float32, self.num_of_genes_to_return, False)
        if self.expr_discretization_method == "Direct_quantile":
            # discretized_expression, zero_expression_genes = discretize_expression_zscores(expression_vector_float32, self.n_bins, std_dev=std_dev_for_another_normal_distr_for_binning)
            discretized_expression, zero_expression_genes = discretize_expression_zscores(expression_vector_float32, self.n_bins)
        elif self.expr_discretization_method == "uniform_bin_count_keep_ones":
            discretized_expression, zero_expression_genes = uniform_bin_count_keep_ones(expression_vector_float32, self.n_bins)
        else:
            print("Unrecognized expr_discretization_method")

        masked_expression, mask = self.bert_masking.mask_sequence(discretized_expression)

        result = {
            "gene_indices": self.gene_indices_for_ret[top_abs_indices],
            "masked_expression": torch.tensor(masked_expression, dtype=torch.int32),
            "true_expression": torch.tensor(discretized_expression, dtype=torch.int32),
            "raw_expression": torch.tensor(expression_vector_float32, dtype=torch.float32),
            "zero_expression_genes": torch.tensor(zero_expression_genes, dtype=torch.bool),
            "masked_booleans": torch.tensor(mask, dtype=torch.bool)
        }
        if self.sort_return_expr_numerically:
            result = self.sort_result_based_on_expr(result, expression_vector_float32)
        return self.prepend_special_tokens(result, self.number_of_special_embeddings)



