import numpy as np
import glob
from math import ceil
import re
from utils.config_loader import Config
import sys
import os
# !!! Please set in command line when qsub to PBS
# param_json_file = '/g/data/yr31/zs2131/tasks/2023/RNA_expr_net/anal/training/exp9/exp9_loss_on_masked_only_8000genes.param_config.json'

# if 'PARAM_JSON_FILE' not in os.environ:
#     os.environ['PARAM_JSON_FILE'] = param_json_file
    
# #os.environ['RUNNING_MODE'] = 'debug'
# os.environ['RUNNING_MODE'] = 'training'

import pandas as pd

config = Config()
proj_path = config.proj_path
ARCHS_file_basename_prefix = config.get("ARCHS_file_basename_prefix")
ARCHS_gene_expression_h5_path = config.get("ARCHS_gene_expression_h5_path")

def read_expression_file(file_path):
    return np.load(file_path)

def split_expression_data(expr_file_path, total_chunks, chunk_idx):
    expression_data = read_expression_file(expr_file_path)
    chunk_size = ceil(expression_data.shape[1] / total_chunks)
    start_idx = chunk_idx * chunk_size
    end_idx = min((chunk_idx + 1) * chunk_size, expression_data.shape[1])
    chunk = np.array(expression_data[:, start_idx:end_idx])
    return chunk




def add_bool_for_nondup_max_z_genes(zscore_filenames, output_file_prefix):
    
    def get_gene_stat_filename(zscore_filename):
        return re.sub(r'_zscore_chunk_(\d+)\.npy', '.gene_stats_chunk_\\1.tsv', zscore_filename)
    gene_stat_filenames = [get_gene_stat_filename(zscore_filename) for zscore_filename in zscore_filenames]
    # Initialize an empty list to store all DataFrames
    df_list = []

# Read each file into a pandas DataFrame and append to the list
    for filename in gene_stat_filenames:
        df = pd.read_csv(filename, sep="\t")
        df_list.append(df)
    
    # Concatenate all the dataframes along the rows
    gene_stat_df = pd.concat(df_list, axis=0)
    
    
    # Add an auxiliary column 'original_order' to keep track of the original order
    gene_stat_df['original_order'] = np.arange(len(gene_stat_df))
    
    # Sort the dataframe based on 'gene_symbol' and 'gene_mean'
    sorted_df = gene_stat_df.sort_values(['gene_symbol', 'gene_mean'], ascending=[True, False])
    
    # Reset the index of sorted_df to create a new one
    sorted_df.reset_index(inplace=True)
    
    # Get the first occurrence of each 'gene_symbol' (which will have the max 'gene_mean' due to sorting)
    max_genes = sorted_df.drop_duplicates(subset='gene_symbol', keep='first')
    
    # Create a new 'max_mean_nondup' column in gene_stat_df and set all its values to False
    gene_stat_df['max_mean_nondup'] = False
    
    # Use 'isin' method to set 'max_mean_nondup' to True for the rows that have the indices in max_genes['original_order']
    gene_stat_df.loc[gene_stat_df['original_order'].isin(max_genes['original_order']), 'max_mean_nondup'] = True
    
    # Sort by 'original_order' to restore the original order
    gene_stat_df.sort_values('original_order', inplace=True)
    
    # Drop the 'original_order' column as it's no longer needed
    gene_stat_df.drop(columns='original_order', inplace=True)
    
    gene_stat_df.to_csv(output_file_prefix + ".gene_stat_filt_on_z_dup.tsv", sep='\t', index=False)

def merge_sample_chunks(output_file_prefix, total_chunks, chunk_idx, idx_genes_of_interest):
    #the files here are chunks of genes
    all_files = glob.glob(f'{output_file_prefix}_zscore_chunk_*.npy')
    all_files.sort(key=lambda x: int(re.search(r'(\d+)(?!.*\d)', x).group(0)))
    #print(all_files)
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
    combined_matrix = combined_matrix[idx_genes_of_interest, :].transpose().astype('float32')

    # Save to a .npy file
    np.save(f'{output_file_prefix}_final_chunk_{chunk_idx}.npy', combined_matrix)
    

# usage
chunk_idx = int(sys.argv[1])
total_chunks = int(sys.argv[2])
use_and_keep_zero_expr_genes = sys.argv[3].lower() == 'true'
h5_file_path = ARCHS_gene_expression_h5_path
# Modify output file prefix based on the third argument
if use_and_keep_zero_expr_genes:
    output_file_prefix = proj_path + f"/data/external/ARCHS/normalize_each_gene/{ARCHS_file_basename_prefix}_with_zero_expr_genes"
else:
    output_file_prefix = proj_path + f"/data/external/ARCHS/normalize_each_gene/{ARCHS_file_basename_prefix}_without_zero_expr_genes"

all_files = glob.glob(f'{output_file_prefix}_zscore_chunk_*.npy')
all_files.sort(key=lambda x: int(re.search(r'(\d+)(?!.*\d)', x).group(0)))
add_bool_for_nondup_max_z_genes(all_files, output_file_prefix)

# Change this to True if would like to output the z-score normalized dataset npy files
if False:
    from data.ARCHSDataset import ARCHSDataset

    archs_dataset = ARCHSDataset(h5_file_path, gene_stat_uniq_bool_file=output_file_prefix + ".gene_stat_filt_on_z_dup.tsv")
    idx_genes_of_interest = archs_dataset.in_gene2vec_nondup_bool
    #output_file_prefix = "/path/to/your/files/"

    # in z-scores were set to 0 for ZERO expression genes in {output_file_prefix}_final_chunk_{chunk_idx}.npy, if use_and_keep_zero_expr_genes == False
    # if use_and_keep_zero_expr_genes == True, zero expression genes will have small (negative) values
    merged_matrix = merge_sample_chunks(output_file_prefix, total_chunks, chunk_idx, idx_genes_of_interest)
