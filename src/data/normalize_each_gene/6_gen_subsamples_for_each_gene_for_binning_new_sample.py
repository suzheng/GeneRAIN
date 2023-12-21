import numpy as np
import glob
from math import ceil
import re
from utils.config_loader import Config
import sys
import os
import argparse
import pandas as pd
# !!! Please set in command line when qsub to PBS
# param_json_file = '/g/data/yr31/zs2131/tasks/2023/RNA_expr_net/anal/training/exp9/exp9_loss_on_masked_only_8000genes.param_config.json'

parser = argparse.ArgumentParser(description='Description of your script.')

# Define named arguments


parser.add_argument('--num_bins', type=int, required=True, help='Total number of expression bins')
parser.add_argument('--subsampling_frac', type=float, required=True, help='Frac for subsampling samples')
parser.add_argument('--use_and_keep_zero_expr_genes', type=str, default='true', choices=['true', 'false'], help='Use and keep zero expression genes')
parser.add_argument('--output_all_genes_in_h5', type=str, default='false', choices=['true', 'false'], help='Do not do any filtering on genes and output all genes in h5')

# Parse the arguments
args = parser.parse_args()


subsampling_frac = args.subsampling_frac
num_bins = args.num_bins
use_and_keep_zero_expr_genes = args.use_and_keep_zero_expr_genes.lower() == 'true'
output_all_genes_in_h5 = args.output_all_genes_in_h5.lower() == 'true'
gene_emb_name = "all_genes"
idx_genes_of_interest = None

config = Config()
proj_path = config.proj_path
ARCHS_file_basename_prefix = config.get("ARCHS_file_basename_prefix")
ARCHS_gene_expression_h5_path = config.get("ARCHS_gene_expression_h5_path")
h5_file_path = ARCHS_gene_expression_h5_path

print(f"\n\nusing gene_emb_name {gene_emb_name}!!\n\n")
if use_and_keep_zero_expr_genes:
    print("Use and keep zero expression genes")
    output_file_prefix = proj_path + f"/data/external/ARCHS/normalize_each_gene/{ARCHS_file_basename_prefix}_with_zero_expr_genes"
else:
    print("Do NOT use and keep zero expression genes")
    output_file_prefix = proj_path + f"/data/external/ARCHS/normalize_each_gene/{ARCHS_file_basename_prefix}_without_zero_expr_genes"

if not output_all_genes_in_h5:
    from train.common_params_funs import *
    from data.ARCHSDataset import ARCHSDataset
    gene_emb_name = GENE_EMB_NAME
    archs_dataset = ARCHSDataset(h5_file_path, gene_stat_uniq_bool_file=output_file_prefix + ".gene_stat_filt_on_z_dup.tsv")
    idx_genes_of_interest = archs_dataset.in_gene2vec_nondup_bool
    #output_file_prefix = "/path/to/your/files/"
    np.savetxt(f'{output_file_prefix}_bin_tot{num_bins}_{gene_emb_name}_{subsampling_frac}_subsampled.gene_symbols.txt', archs_dataset.gene_symbols_in_npy_file, fmt='%s')

def subsample_samples_for_each_genes(genes_by_samples_mat, frac=0.01):

    # Step 1: Sort the expression values for each gene
    sorted_matrix = np.sort(genes_by_samples_mat, axis=1)

    # Step 2: Subsample at 1% intervals
    subsample_indices = np.linspace(0, sorted_matrix.shape[1] - 1, int(frac * sorted_matrix.shape[1])).astype(int)
    subsampled_matrix = sorted_matrix[:, subsample_indices]
    return subsampled_matrix

def merge_sample_chunks(output_file_prefix, idx_genes_of_interest, num_bins, subsampling_frac=0.005):
    #the files here are chunks of genes
    all_files = glob.glob(f'{output_file_prefix}_normalized_logged.genechunk_*.npy')
    all_files.sort(key=lambda x: int(re.search(r'(\d+)(?!.*\d)', x).group(0)))
    print(all_files)
    combined_matrix = None
    # each file is a gene_by_sample_mat, files are split by genes. Each file is a chunk of genes for all the samples.
    for file_path in all_files:
        print(f"processing {file_path}")
        #only get one chunk of samples
        ori_genes_by_samples_mat =  np.load(file_path)
        subsampled_genes_by_samples_mat = subsample_samples_for_each_genes(ori_genes_by_samples_mat, subsampling_frac)
        if combined_matrix is None:
            combined_matrix = subsampled_genes_by_samples_mat
        else:
            combined_matrix = np.vstack((combined_matrix, subsampled_genes_by_samples_mat))
        # Transpose and convert to float32
    print(f"shape of unfiltered chunk {combined_matrix.shape}")
    #print(f"length of idx_genes_of_interest {len(idx_genes_of_interest)}")
    if idx_genes_of_interest == None:
        combined_matrix = combined_matrix.transpose().astype('float32')
        print("output all genes")
    else:
        combined_matrix = combined_matrix[idx_genes_of_interest, :].transpose().astype('float32')
        print("output selected genes")

    print(f"shape of filtered chunk {combined_matrix.shape}")
    # Save to a .npy file
    np.save(f'{output_file_prefix}_bin_tot{num_bins}_{gene_emb_name}_{subsampling_frac}_subsampled.npy', combined_matrix)
    print(f'saved to {output_file_prefix}_bin_tot{num_bins}_{gene_emb_name}_{subsampling_frac}_subsampled.npy')
    
merged_matrix = merge_sample_chunks(output_file_prefix, idx_genes_of_interest, num_bins, subsampling_frac)
