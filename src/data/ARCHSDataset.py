from utils.params import params
params = params()
import h5py
import torch
import numpy as np
import pandas as pd
import sys
from torch.utils.data import Dataset
from data.BertMasking import get_bert_masking

from utils.json_utils import JsonUtils
ju = JsonUtils()
from train.common_params_funs import config, get_gene2idx, get_gene2idx_of_whole_gene_emb
from train.common import train


from utils.config_loader import Config
config = Config()
proj_path = config.proj_path
ARCHS_file_basename_prefix = config.get("ARCHS_file_basename_prefix")
ARCHS_gene_expression_h5_path = config.get("ARCHS_gene_expression_h5_path")
from data.discretize_expression import discretize_expression_zscores, discretize_expression_uniform
from data.discretize_expression import uniform_bin_count_keep_ones
from data.data_utils import get_top_genes

# Load gene_to_idx dictionary from JSON file
# get_gene2idx will get the genes satisfying nondup_qualified & mean_z_scores
# get_gene2idx_of_whole_gene_emb will get all the genes without filtering for nondup_qualified & mean_z_scores
# 5_combine_genes_filter_genes_split_into_sample_chunks_for_bins.py use attribute of in_gene2vec_nondup_bool to filter genes to create dataset. 
# In coding_lncrna dataset, only genes satisfying nondup_qualified & mean_z_scores are included

# To keep the historical code consistent
if params.GENE_EMB_NAME == "gene2vec":
    gene_to_idx = get_gene2idx_of_whole_gene_emb()
else:
    gene_to_idx, _ = get_gene2idx()

# std_dev_for_another_normal_distr_for_binning = STD_DEV_FOR_ANOTHER_NORMAL_DISTR_FOR_BINNING
use_and_keep_zero_expr_genes = params.USE_AND_KEEP_ZERO_EXPR_GENES
if use_and_keep_zero_expr_genes:
    output_file_prefix = proj_path + f"/data/external/ARCHS/normalize_each_gene/{ARCHS_file_basename_prefix}_with_zero_expr_genes"
else:
    output_file_prefix = proj_path + f"/data/external/ARCHS/normalize_each_gene/{ARCHS_file_basename_prefix}_without_zero_expr_genes"

# None of dataset will have special tokens, all special tokens will be added outside of GeneExprTransformer Module, inside the wrapper modules.

class ARCHSDataset(Dataset):
    def __init__(self, 
    h5_file_path=ARCHS_gene_expression_h5_path, 
    n_bins=params.NUM_BINS, 
    mask_fraction=params.MASK_FRACTIONS[0], 
    expr_discretization_method=params.EXPR_DISCRETIZATION_METHOD, 
    load_data_into_mem=False, 
    chunk_idx=0, 
    num_of_genes=params.NUM_OF_GENES_SELECTED,
    gene_stat_uniq_bool_file=output_file_prefix + ".gene_stat_filt_on_z_dup.tsv",
    min_mean_val_for_zscore=params.MIN_MEAN_VAL_FOR_ZSCORE,
    only_use_postive_zscores_in_training=params.ONLY_USE_POSITIVE_ZSCORES_IN_TRAINING,
    number_of_special_embeddings=params.NUMBER_OF_SPECIAL_TOKEN_IN_DATASET,
    sort_return_expr_numerically=(params.TRANSFORMER_MODEL_NAME == "GPT" or params.TRANSFORMER_MODEL_NAME == "Bert_pred_tokens")
    ):
        self.h5_file_path = h5_file_path
        self.h5_file = h5py.File(self.h5_file_path, "r")
        self.gene_symbols = self.h5_file['meta']['genes']['symbol'][()]
        self.sample_size = self.h5_file['data']['expression'].shape[1]
        self.n_bins = n_bins
        self.num_of_genes_to_return = num_of_genes
        self.sort_return_expr_numerically = sort_return_expr_numerically
        # stat for the mean and std for each gene, and a boolean to indicate if that genes should be selected 
        # to ensure the uniqueness
        self.gene_stat_uniq_bool_file = gene_stat_uniq_bool_file
        self.min_mean_val_for_zscore = min_mean_val_for_zscore
        self.only_use_postive_zscores_in_training = only_use_postive_zscores_in_training
        # Find the index of genes in gene_to_idx
        self.gene_symbol_indices = np.array([gene_to_idx.get(gene_symbol.decode('utf-8'), -1) for gene_symbol in self.gene_symbols], dtype=np.int32)
        self.mask_fraction = mask_fraction
        # Filter gene_symbol_indices to keep only the genes with index in gene_to_idx
        self.genes_in_gene2vec = self.gene_symbol_indices != -1
        self.number_of_special_embeddings = number_of_special_embeddings
        # we put it here, as we would like to make treating it a hyperparam easier, 
        # we don't have to go back to generate the gene_stat_uniq_bool_file file, if we do the filtering, 
        # before generating that file. We would like to remove the genes that have very small means,
        # which can make the normally expressed genes in this dataset have very high z-score
        self.non_dup_gene_bool, self.qualified_mean_bool = self.get_non_dup_and_qualified_mean_bool_vectors()
        self.in_gene2vec_nondup_bool = self.genes_in_gene2vec & self.non_dup_gene_bool
        self.in_gene2vec_nondup_qualified_mean_bool = self.in_gene2vec_nondup_bool & self.qualified_mean_bool
        self.bert_masking = get_bert_masking(mask_fraction=self.mask_fraction)

        # zscore final npy files only have in_gene2vec_nondup_bool, but no qualified_mean was applied
        # so length of self.qualified_mean_bool_for_npy_file will match with the dimension of final npy file
        self.qualified_mean_bool_for_npy_file = self.in_gene2vec_nondup_qualified_mean_bool[self.in_gene2vec_nondup_bool]
        self.filtered_indices = np.where(self.in_gene2vec_nondup_qualified_mean_bool)[0]
        self.gene_indices_for_ret = torch.tensor(self.gene_symbol_indices[self.in_gene2vec_nondup_qualified_mean_bool], dtype=torch.int32)
        #raw integer read count, genes are rows, samples are columns
        self.h5_file_data_expr = self.h5_file['data']['expression']
        self.expr_discretization_method = expr_discretization_method
        self.load_data_into_mem = load_data_into_mem
        self.gene_symbols_in_npy_file = [gene.decode('utf-8') for gene in self.gene_symbols[self.in_gene2vec_nondup_bool]]
        if load_data_into_mem == True:
            #the rows are samples, it is transposed from the original matrix
            #zscore final npy files only have in_gene2vec_nondup_bool, but no qualified_mean was applied
            
            if self.expr_discretization_method == "Bins_by_gene":
                print("\n\nexpr_discretization_method of Bins_by_gene is not recommended, only when all genes are used can be considered!\n\n")
                print(f'loading {output_file_prefix}_bin_tot100_final_chunk_{chunk_idx}.npy')
                self.filtered_expression_data = np.load(f'{output_file_prefix}_bin_tot100_final_chunk_{chunk_idx}.npy')
            elif self.expr_discretization_method == "uniform_bin_count_keep_ones":
                print(f'loading {output_file_prefix}_bin_tot2000_final_{params.GENE_EMB_NAME}_chunk_{chunk_idx}.npy')
                self.filtered_expression_data = np.load(f'{output_file_prefix}_bin_tot2000_final_{params.GENE_EMB_NAME}_chunk_{chunk_idx}.npy')
            else:
                print(output_file_prefix + f"_final_chunk_{chunk_idx}.npy")
                self.filtered_expression_data = np.load(output_file_prefix + f"_final_chunk_{chunk_idx}.npy")
            
    def get_non_dup_and_qualified_mean_bool_vectors(self):
        # Read the TSV file
        df = pd.read_csv(self.gene_stat_uniq_bool_file, sep='\t')

        # Create a numpy boolean vector from the 'max_mean_nondup' column
        max_mean_nondup_vector = df['max_mean_nondup'].values.astype(bool)

        # Create a numpy boolean vector that indicates whether the 'gene_mean' values are greater than min_mean_val_for_zscore
        gene_mean_vector = (df['gene_mean'] > self.min_mean_val_for_zscore).values
        return max_mean_nondup_vector, gene_mean_vector
        
    def __len__(self):
        if self.load_data_into_mem:
            return self.filtered_expression_data.shape[0]
        else:        
            return self.sample_size
    
    def get_ori_exp(self, idx):
        expression_vector = self.h5_file_data_expr[self.filtered_indices, idx]
        return expression_vector
    
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
        #zscore final npy files only have in_gene2vec_nondup_bool, but no qualified_mean was applied
        expression_vector = self.filtered_expression_data[idx, self.qualified_mean_bool_for_npy_file]

        expression_vector_float32 = expression_vector.astype(np.float32)

        expression_vector_float32, top_abs_indices = get_top_genes(expression_vector_float32, self.num_of_genes_to_return, self.only_use_postive_zscores_in_training)
        if self.expr_discretization_method == "Direct_quantile":
            # discretized_expression, zero_expression_genes = discretize_expression_zscores(expression_vector_float32, self.n_bins, std_dev=std_dev_for_another_normal_distr_for_binning)
            discretized_expression, zero_expression_genes = discretize_expression_zscores(expression_vector_float32, self.n_bins)
        elif self.expr_discretization_method == "Bins_by_gene":
            discretized_expression = expression_vector_float32
            zero_expression_genes = np.zeros_like(expression_vector_float32, dtype=bool)
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

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()

