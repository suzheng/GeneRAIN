from utils.params import params
params = params()
import argparse
import anndata
import numpy as np
import pandas as pd
import pickle
from data.adata import Adata
from utils.config_loader import Config
import os
from data.GetBinsByGeneForNewSamples import get_bins_by_gene_for_new_samples

parser = argparse.ArgumentParser(description='Process dataset label argument')

parser.add_argument('--dataset_label', type=str, required=True, help='Label for the dataset (e.g., K562_essential)')
parser.add_argument('--use_zero_expr_genes', type=str, default='true', help='Whether to use zero expression genes. Expected values are "true" or "false".')

args = parser.parse_args()

use_zero_expr_genes = args.use_zero_expr_genes.lower() == 'true'

config = Config()

os.environ['RUNNING_MODE'] = 'training'

from train.common_params_funs import config
from train.common import train


gene_emb_name = params.GENE_EMB_NAME
dataset_label = args.dataset_label

file_path_key_in_config=f"fine_tuning_{dataset_label}_dataset_file_path"
dataset_file = config.get(file_path_key_in_config)

output_pickle_file = dataset_file.replace("mean_agg.pickle", f"mean_agg.{gene_emb_name}.binned.pickle")
output_tsv_file = dataset_file.replace("mean_agg.pickle", f"mean_agg.{gene_emb_name}.binned.tsv")
samples_by_genes_subsampled_file = config.proj_path + "/data/external/ARCHS/normalize_each_gene/" + config.get("ARCHS_file_basename_prefix") + f"_with_zero_expr_genes_bin_tot2000_{gene_emb_name}_0.005_subsampled.npy"
subsampled_gene_symbol_file = config.proj_path + "/data/external/ARCHS/normalize_each_gene/" + config.get("ARCHS_file_basename_prefix") + f"_with_zero_expr_genes_bin_tot2000_{gene_emb_name}_0.005_subsampled.gene_symbols.txt"

if use_zero_expr_genes == False:
    print("NOT use zero expression genes")
    output_pickle_file = dataset_file.replace("mean_agg.pickle", f"mean_agg.without_zero_expr_genes.{gene_emb_name}.binned.pickle")
    output_tsv_file = dataset_file.replace("mean_agg.pickle", f"mean_agg.without_zero_expr_genes.{gene_emb_name}.binned.tsv")
    samples_by_genes_subsampled_file = config.proj_path + "/data/external/ARCHS/normalize_each_gene/" + config.get("ARCHS_file_basename_prefix") + f"_without_zero_expr_genes_bin_tot2000_{gene_emb_name}_0.005_subsampled.npy"
    subsampled_gene_symbol_file = config.proj_path + "/data/external/ARCHS/normalize_each_gene/" + config.get("ARCHS_file_basename_prefix") + f"_without_zero_expr_genes_bin_tot2000_{gene_emb_name}_0.005_subsampled.gene_symbols.txt"

with open(dataset_file, 'rb') as f:
    adata = pickle.load(f)
#print(dataset_file)
cell_group_names = adata.obs_names
gene_names = adata.var_names
sample_by_gene_expr_matrix = adata.X



gene_by_sample_mat_binned, bool_for_if_samples_included_in_returned_mat, gene_exist_in_subsampled, stats_df = get_bins_by_gene_for_new_samples(samples_by_genes_subsampled_file, 
                                 subsampled_gene_symbol_file, 
                                 gene_by_sample_expr_mat=sample_by_gene_expr_matrix.T, 
                                 quantified_gene_list=gene_names, 
                                 output_prefix=None, 
                                 num_gene_bin = 2000,  
                                 min_total_count=-1)

binned_adata = Adata(np.array(cell_group_names)[bool_for_if_samples_included_in_returned_mat], 
                     np.array(gene_names)[gene_exist_in_subsampled],
                     gene_by_sample_mat_binned.T[:,gene_exist_in_subsampled]
                    )
with open(output_pickle_file, 'wb') as f:
    pickle.dump(binned_adata, f)
print(f"expr data saved to {output_pickle_file}!")

binned_adata.save_matrix_to_tsv(output_tsv_file)
print(f'Data saved to {output_tsv_file}')



