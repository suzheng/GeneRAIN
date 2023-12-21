import numpy as np
import pandas as pd
import sys
import h5py
from math import ceil
from utils.config_loader import Config

config = Config()
proj_path = config.proj_path
ARCHS_file_basename_prefix = config.get("ARCHS_file_basename_prefix")
ARCHS_gene_expression_h5_path = config.get("ARCHS_gene_expression_h5_path")

# in z-scores were set to 0 for ZERO expression genes in {output_file_prefix}_final_chunk_{chunk_idx}.npy, if use_and_keep_zero_expr_genes == False
# if use_and_keep_zero_expr_genes == True, zero expression genes will have small (negative) values
def process_gene_chunk(gene_chunk_index, total_chunks, h5_file_path, output_file_prefix, use_and_keep_zero_expr_genes=False):
    print(f"Processing gene chunk {gene_chunk_index}")
    zscore_output_file = f'{output_file_prefix}_zscore_chunk_{gene_chunk_index}.npy'
    #zero_expr_output_file = f'{output_file_prefix}_zscore_chunk_{gene_chunk_index}.zero_expr.npy'
    
    # Open the h5 file and get gene symbols
    with h5py.File(h5_file_path, "r") as h5_file:
        # Get gene symbols
        gene_symbols = h5_file['meta']['genes']['symbol'][()]
        num_genes = len(gene_symbols)
    
    gene_chunk_size = ceil(num_genes / total_chunks)

    # Calculate start and end indices of genes for this chunk
    start_index = gene_chunk_index * gene_chunk_size
    end_index = min((gene_chunk_index + 1) * gene_chunk_size, num_genes)
    
    # Get the gene symbols for this chunk
    gene_symbols_chunk = gene_symbols[start_index:end_index]
    gene_symbols_chunk = [symbol.decode() for symbol in gene_symbols_chunk]
    # Initialize a container for z-scores of this chunk
    z_scores_chunk = np.empty((end_index - start_index, 0))
    samples = []
    
    # Read and process each sample chunk file
    sample_chunk_index = 0
    while True:
        try:


            # Load the sample chunk file
            sample_chunk_file = proj_path + f'/data/external/ARCHS/normalize_each_gene/{ARCHS_file_basename_prefix}.h5.sc_filtered_chunk_{sample_chunk_index}.npy'
            sample_chunk = np.load(sample_chunk_file)
            print(f"loaded {sample_chunk_file}")
            # Extract the genes of interest
            genes_of_interest = sample_chunk[start_index:end_index, :]
            del sample_chunk
            # Append the genes to the samples list
            samples.append(genes_of_interest)
            del genes_of_interest
            sample_chunk_index += 1
        except FileNotFoundError:
            break

    # Concatenate all samples
    all_samples = np.concatenate(samples, axis=1)
    print(f"all_samples shape {all_samples.shape}")
    # Calculate mean and standard deviation for z-score normalization
    if use_and_keep_zero_expr_genes:
        masked_expression = np.ma.masked_equal(all_samples, -1)
    else:
        masked_expression = np.ma.masked_equal(all_samples, 0)
    del all_samples
    gene_mean = np.ma.mean(masked_expression, axis=1)
    gene_std = np.ma.std(masked_expression, axis=1)
    zero_expr = masked_expression.mask
    del masked_expression
    print("calculated mean and sd")
    # Iterate over each sample to calculate z-scores
    for sample in samples:
        print(f"processing sample with shape {sample.shape}")
        # Calculate the z-score for each gene
        z_score_expression = (sample - gene_mean[:, np.newaxis]) / gene_std[:, np.newaxis]
        
        # Replace any NaNs or Infs that might result from division by zero
        z_score_expression = np.nan_to_num(z_score_expression)
        
        # Append the z-scores to the chunk container
        z_scores_chunk = np.hstack((z_scores_chunk, z_score_expression))
        del z_score_expression
    z_scores_chunk[zero_expr] = 0
    # Save the z-score data
    np.save(zscore_output_file, z_scores_chunk.data.astype(np.float32))
    #np.save(zero_expr_output_file, zero_expr)

    return gene_symbols_chunk, gene_mean, gene_std

if __name__ == "__main__":
    gene_chunk_index = int(sys.argv[1])
    total_chunks = int(sys.argv[2])
    use_and_keep_zero_expr_genes = sys.argv[3].lower() == 'true'
    h5_file_path = ARCHS_gene_expression_h5_path
    # Modify output file prefix based on the third argument
    if use_and_keep_zero_expr_genes:
        output_file_prefix = proj_path + f"/data/external/ARCHS/normalize_each_gene/{ARCHS_file_basename_prefix}_with_zero_expr_genes"
    else:
        output_file_prefix = proj_path + f"/data/external/ARCHS/normalize_each_gene/{ARCHS_file_basename_prefix}_without_zero_expr_genes"
    
    gene_symbols_chunk, gene_mean, gene_std = process_gene_chunk(gene_chunk_index, total_chunks, h5_file_path, output_file_prefix, use_and_keep_zero_expr_genes)
    
    # Save the gene stats
    output_gene_stats_file = output_file_prefix + f'.gene_stats_chunk_{gene_chunk_index}.tsv'
    gene_stats_df = pd.DataFrame({
        "gene_symbol": gene_symbols_chunk,
        "gene_mean": gene_mean,
        "gene_std": gene_std
    })
    gene_stats_df.to_csv(output_gene_stats_file, sep="\t", index=False)
