import numpy as np
import h5py
import argparse
import pandas as pd
from math import ceil
from utils.config_loader import Config

config = Config()
proj_path = config.proj_path
ARCHS_file_basename_prefix = config.get("ARCHS_file_basename_prefix")
ARCHS_gene_expression_h5_path = config.get("ARCHS_gene_expression_h5_path")

def split_expression_data_old(h5_file_path, total_chunks, chunk_idx):
    with h5py.File(h5_file_path, "r") as h5_file:
        expression_data = h5_file['data']['expression']
        chunk_size = ceil(expression_data.shape[1] / total_chunks)
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, expression_data.shape[1])
        chunk = np.array(expression_data[:, start_idx:end_idx])
        return chunk
        #np.save(f'{output_prefix}_chunk_{chunk_idx}.npy', chunk)

def process_chunk_old(expression_data, output_prefix, min_total_count=1000000, sc_prob_thres=0.5, min_expr_gene=1000):
    #print(f"Processing {input_file}")
    #expression_data = np.load(input_file)
    total_counts = np.sum(expression_data, axis=0)
    valid_samples = total_counts >= min_total_count
    expression_data = expression_data[:, valid_samples]
    total_counts = total_counts[valid_samples]
    scaling_factors = 10000000 / total_counts
    normalized_expression = expression_data * scaling_factors[np.newaxis, :]
    log_transformed_expression = np.log10(normalized_expression + 1)
    np.save(f'{output_prefix}.npy', log_transformed_expression.astype(np.float32))
    pd.DataFrame(valid_samples).to_csv(f'{output_prefix}_valid_samples.tsv', sep='\t', header=False, index=True)

def split_expression_data(h5_file_path, total_chunks, chunk_idx):
    with h5py.File(h5_file_path, "r") as h5_file:
        expression_data = h5_file['data']['expression']
        meta_data = h5_file['meta']['samples']

        chunk_size = ceil(expression_data.shape[1] / total_chunks)
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, expression_data.shape[1])

        expression_chunk = np.array(expression_data[:, start_idx:end_idx])

        meta_data_chunk = {key: np.array(meta_data[key][start_idx:end_idx]) for key in meta_data.keys()}

    # Convert bytes to string for metadata
    meta_data_chunk = {key: [x.decode() if isinstance(x, bytes) else x for x in value] 
                       for key, value in meta_data_chunk.items()}

    return expression_chunk, meta_data_chunk


def process_chunk(expression_data, meta_data_chunk, output_prefix, min_total_count=1000000, sc_prob_thres=0.5, min_expr_genes=2000):
    #print(f"Processing {input_file}")
    #expression_data = np.load(input_file)
    sc_prob = np.array(meta_data_chunk["singlecellprobability"]).astype(float)
    total_counts = np.sum(expression_data, axis=0)
    non_zero_counts = np.count_nonzero(expression_data, axis=0)
    mean_counts = np.mean(expression_data, axis=0)
    median_counts = np.median(expression_data, axis=0)
    max_counts = np.max(expression_data, axis=0)
    valid_samples = (total_counts >= min_total_count) & \
                (sc_prob < sc_prob_thres) & \
                (non_zero_counts > min_expr_genes)
    expression_data = expression_data[:, valid_samples]
    total_counts_valid = total_counts[valid_samples]
    scaling_factors = 10000000 / total_counts_valid
    normalized_expression = expression_data * scaling_factors[np.newaxis, :]
    log_transformed_expression = np.log10(normalized_expression + 1)
    np.save(f'{output_prefix}.npy', log_transformed_expression.astype(np.float32))
    # Save the valid sample flags
    #pd.DataFrame(valid_samples).to_csv(f'{output_prefix}_valid_samples.tsv', sep='\t', header=False, index=True)

    # Save other stats in a dataframe
    stats_df = pd.DataFrame({
        'total_read_counts': total_counts,
        'non_zero_genes': non_zero_counts,
        'mean_read_counts': mean_counts,
        'median_read_counts': median_counts,
        'max_read_counts': max_counts,
        'valid_samples': valid_samples
    })
    
    # Save the stats dataframe to a TSV file
    stats_df.to_csv(f'{output_prefix}_stats.tsv', sep='\t', header=True, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split and save a specific chunk of expression data.')
    parser.add_argument('--total_chunks', type=int, required=True, help='The total number of chunks to split the data into.')
    parser.add_argument('--chunk_idx', type=int, required=True, help='The index of the chunk to save.')
    parser.add_argument('--min_total_count', type=int, default=1000000, help='Minimum total read count for filtering samples.')
    parser.add_argument('--sc_prob_thres', type=float, default=0.5, help='Samples with singlecellprobability greater than this value will be filtered out.')
    parser.add_argument('--min_expr_genes', type=float, default=2000, help='Minimum number genes with non-zero expression for filtering samples. Please note that the total number of genes in the matrix is over 50,000')
    args = parser.parse_args()

    # h5_file_path = proj_path + "/data/external/ARCHS/archs4_gene_human_v2.1.2.h5"
    # output_prefix = proj_path + f"/data/external/ARCHS/normalize_each_gene/archs4_gene_human_v2.1.2.samples_filtered_chunk_{args.chunk_idx}"

    h5_file_path = ARCHS_gene_expression_h5_path
    output_prefix = proj_path + f"/data/external/ARCHS/normalize_each_gene/{ARCHS_file_basename_prefix}.h5.sc_filtered_chunk_{args.chunk_idx}"

    expression_data, meta_data_chunk = split_expression_data(h5_file_path, args.total_chunks, args.chunk_idx)
    process_chunk(expression_data, meta_data_chunk, output_prefix, min_total_count=args.min_total_count, sc_prob_thres=args.sc_prob_thres, min_expr_genes=args.min_expr_genes)


