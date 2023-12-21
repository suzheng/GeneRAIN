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

# if 'PARAM_JSON_FILE' not in os.environ:
#     os.environ['PARAM_JSON_FILE'] = param_json_file
    
# #os.environ['RUNNING_MODE'] = 'debug'
# os.environ['RUNNING_MODE'] = 'training'
# usage
# Create the ArgumentParser object
parser = argparse.ArgumentParser(description='Description of your script.')

# Define named arguments
parser.add_argument('--chunk_idx', type=int, required=True, help='Chunk index')
parser.add_argument('--total_chunks', type=int, required=True, help='Total number of chunks')
parser.add_argument('--num_bins', type=int, required=True, help='Total number of expression bins')
parser.add_argument('--use_and_keep_zero_expr_genes', type=str, default='true', choices=['true', 'false'], help='Use and keep zero expression genes')
parser.add_argument('--output_all_genes_in_h5', type=str, default='false', choices=['true', 'false'], help='Do not do any filtering on genes and output all genes in h5')

# Parse the arguments
args = parser.parse_args()

# Access arguments
chunk_idx = args.chunk_idx
total_chunks = args.total_chunks
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



def read_expression_file(file_path):
    return np.load(file_path)

def split_expression_data(expr_file_path, total_chunks, chunk_idx):
    expression_data = read_expression_file(expr_file_path)
    chunk_size = ceil(expression_data.shape[1] / total_chunks)
    start_idx = chunk_idx * chunk_size
    end_idx = min((chunk_idx + 1) * chunk_size, expression_data.shape[1])
    chunk = np.array(expression_data[:, start_idx:end_idx])
    return chunk


def merge_sample_chunks(output_file_prefix, total_chunks, chunk_idx, idx_genes_of_interest, num_bins):
    #the files here are chunks of genes
    all_files = glob.glob(f'{output_file_prefix}_bin_tot{num_bins}.chunk_*.npy')
    all_files.sort(key=lambda x: int(re.search(r'(\d+)(?!.*\d)', x).group(0)))
    print(all_files)
    combined_matrix = None
    for file_path in all_files:
        print(f"processing {file_path}")
        #only get one chunk of samples
        chunk = split_expression_data(file_path, total_chunks, chunk_idx)
        if combined_matrix is None:
            combined_matrix = chunk
        else:
            combined_matrix = np.vstack((combined_matrix, chunk))
        # Transpose and convert to float32
    print(f"shape of unfiltered chunk {combined_matrix.shape}")
    if idx_genes_of_interest == None:
        combined_matrix = combined_matrix.transpose().astype('float32')
        print("output all genes")
    else:
        combined_matrix = combined_matrix[idx_genes_of_interest, :].transpose().astype('float32')
        print("output selected genes")

    # Save to a .npy file
    np.save(f'{output_file_prefix}_bin_tot{num_bins}_final_{gene_emb_name}_chunk_{chunk_idx}.npy', combined_matrix)
    print(f"Saved to {output_file_prefix}_bin_tot{num_bins}_final_{gene_emb_name}_chunk_{chunk_idx}.npy!")
    
#output_file_prefix = "/path/to/your/files/"

merged_matrix = merge_sample_chunks(output_file_prefix, total_chunks, chunk_idx, idx_genes_of_interest, num_bins)
