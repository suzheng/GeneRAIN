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

# zeros stay in bin 1!
def cal_bin_for_each_gene_in_gene_by_sample_mat(masked_expression, num_bins):
# Create a mask where the expressions are non-zero
    # Create a mask where the expressions are non-zero
    # this matrix is log-tranformed
    non_zero_mask = (masked_expression > 0)
    
    # Rank entire matrix; zeros will get the lowest ranks
    full_ranks = np.argsort(np.argsort(masked_expression, axis=1), axis=1)
    
    # Find out how many zeros are in each row (gene)
    num_zeros = masked_expression.shape[1] - np.sum(non_zero_mask, axis=1)
    
    # Subtract the number of zeros from non-zero ranks
    adjusted_ranks = full_ranks - num_zeros[:, np.newaxis] * non_zero_mask
    
    # Only take non-zero values for further processing
    adjusted_ranks = adjusted_ranks * non_zero_mask
    
    # Calculate maximum rank per row (gene)
    max_ranks = np.max(adjusted_ranks, axis=1)
    
    # We only adjust bin size for genes that have non-zero expressions
    bin_sizes = np.where(max_ranks > 0, max_ranks / (num_bins - 1), 1)
    
    # Convert ranks to bins; this will ensure zeros stay 1
    bins = (adjusted_ranks / bin_sizes[:, np.newaxis]).astype(int) + 1 + 1*non_zero_mask
    
    # Make sure bins do not exceed 100
    bins[bins > num_bins] = num_bins
    
    # Assign this to our binned_rank
    binned_rank = bins
    
    # Combine masked_expression and binned_rank
    #combined_matrix = np.vstack((masked_expression.data, binned_rank))
    
    # Save the combined matrix to a TSV file
    #np.savetxt('combined_matrix2.tsv', combined_matrix.T, delimiter='\t', fmt='%s')
    return binned_rank

def process_gene_chunk(gene_chunk_index, total_chunks, h5_file_path, output_file_prefix, num_bins):
    print(f"Processing gene chunk {gene_chunk_index}")
    normalized_logged_outfile = f'{output_file_prefix}_normalized_logged.genechunk_{gene_chunk_index}.npy'
    bin_output_file = f'{output_file_prefix}_bin_tot{num_bins}.chunk_{gene_chunk_index}.npy'
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
            # if sample_chunk_index > 1:
            #     break

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
    np.save(normalized_logged_outfile, all_samples)

    all_samples_binned = cal_bin_for_each_gene_in_gene_by_sample_mat(all_samples, num_bins)
    np.save(bin_output_file, all_samples_binned.astype(np.int32))
    print(f"saved to {bin_output_file}")


    return gene_symbols_chunk

if __name__ == "__main__":
    """
    This script take the files generated by 1_split_expr_data_by_sample.py as input, and process one chunk of genes.
    """
    gene_chunk_index = int(sys.argv[1])
    total_chunks = int(sys.argv[2])
    # total number of expr bins
    num_bins = int(sys.argv[3])
    # use_and_keep_zero_expr_genes = True
    use_and_keep_zero_expr_genes = True if len(sys.argv) < 5 else sys.argv[4].lower() == 'true'
    h5_file_path = ARCHS_gene_expression_h5_path
    # Modify output file prefix based on the third argument
    if use_and_keep_zero_expr_genes:
        print("Use and keep zero expression genes")
        output_file_prefix = proj_path + f"/data/external/ARCHS/normalize_each_gene/{ARCHS_file_basename_prefix}_with_zero_expr_genes"
    else:
        print("Do NOT use and keep zero expression genes")
        output_file_prefix = proj_path + f"/data/external/ARCHS/normalize_each_gene/{ARCHS_file_basename_prefix}_without_zero_expr_genes"
    
    gene_symbols_chunk= process_gene_chunk(gene_chunk_index, total_chunks, h5_file_path, output_file_prefix, num_bins)
    
