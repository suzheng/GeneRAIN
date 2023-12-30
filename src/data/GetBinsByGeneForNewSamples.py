
import numpy as np
import pandas as pd
from utils.config_loader import Config
from utils.Ensembl_ID_gene_symbol_mapping_reader import read_ensembl_to_gene_mapping
from utils.json_utils import JsonUtils
import importlib
module_name = "data.normalize_each_gene.4_cal_bins_on_one_chunk_of_genes"
module = importlib.import_module(module_name)
cal_bin_for_each_gene_in_gene_by_sample_mat = getattr(module, "cal_bin_for_each_gene_in_gene_by_sample_mat")

# from train.common_params_funs import config
from train.common import train

config = Config()
ju = JsonUtils()

def get_bins_by_gene_for_new_samples(samples_by_genes_subsampled_file, subsampled_gene_symbol_file, gene_by_sample_expr_mat, quantified_gene_list, output_prefix=None, num_gene_bin = 2000,  min_total_count=-1):
    """
    Process and bin the gene expression data of the provided matrix using a subsampled data matrix.

    Parameters:
    - samples_by_genes_subsampled_file (str): File containing the subsampled genes by samples matrix.
    - subsampled_gene_symbol_file (str): File containing gene symbols corresponding to the rows of the subsampled matrix.
    - gene_by_sample_expr_mat (numpy.ndarray): Gene by sample expression matrix to be processed and binned.
    - quantified_gene_list (list or numpy.ndarray): List of gene symbols corresponding to the rows of gene_by_sample_expr_mat.
    - output_prefix (str, optional): Prefix for output files such as statistics dataframe. None for no output.
    - num_gene_bin (int, optional): The number of bins to be used for discretization. Defaults to 2000.
    - min_total_count (int, optional): Minimum total count to consider a sample as valid. Defaults to -1.

    Returns:
    - gene_by_sample_mat_binned (numpy.ndarray): Binned version of the input gene_by_sample_expr_mat.
    - bool_for_if_samples_included_in_returned_mat (numpy.ndarray): Boolean array indicating if the samples in the input matrix are included in the processed matrix.
    - gene_exist_in_subsampled (numpy.ndarray): Boolean array indicating which genes from the input matrix are present in the subsampled matrix.
    - stats_df (pandas.DataFrame): DataFrame containing statistics for each sample in the input matrix.

    Notes:
    The function uses a subsampled gene expression dataset to bin the input gene expression dataset. The binned values correspond to the discretized expression values of the genes, based on the subsampled data.
    """
    quantified_gene_list = np.array(quantified_gene_list)
    genes_by_samples_subsampled_mat, subsampled_gene_symbols = get_subsampled_mat_genes(samples_by_genes_subsampled_file, subsampled_gene_symbol_file)
    
    # only valid samples are returned as gene_by_sample_expr_mat_loged. The shape of gene_by_sample_expr_mat and gene_by_sample_expr_mat_loged are different
    gene_by_sample_expr_mat_loged, bool_for_if_samples_included_in_returned_mat, stats_df = scale_by_sample_total_count_log_transform(gene_by_sample_expr_mat, output_prefix=output_prefix, min_total_count=min_total_count)
    
    matched_subsampled_mat, gene_exist_in_subsampled = match_genes_get_subsampled_mat(gene_by_sample_expr_mat_loged, quantified_gene_list, genes_by_samples_subsampled_mat, subsampled_gene_symbols)

    gene_by_sample_mat_binned = cal_bins_for_target_matrix(gene_by_sample_expr_mat_loged, gene_exist_in_subsampled, matched_subsampled_mat, num_gene_bin)
    
    return gene_by_sample_mat_binned, bool_for_if_samples_included_in_returned_mat, gene_exist_in_subsampled, stats_df


def get_subsampled_mat_genes(samples_by_genes_subsampled_file, subsampled_gene_symbol_file):
    genes_by_samples_subsampled_mat = (np.load(samples_by_genes_subsampled_file)).T
    subsampled_gene_symbols = np.loadtxt(subsampled_gene_symbol_file, dtype=str)
    print(f"genes_by_samples_subsampled_mat shape {genes_by_samples_subsampled_mat.shape}")
    if len(subsampled_gene_symbols) != genes_by_samples_subsampled_mat.shape[0]:
        raise ValueError("The number of subsampled gene symbols does not match the number of samples in the subsampled matrix.")
    return genes_by_samples_subsampled_mat, subsampled_gene_symbols


def match_genes_get_subsampled_mat(target_gene_by_sample_mat, genes_in_target_mat, genes_by_samples_subsampled_mat, subsampled_gene_symbols):
    """
    Match the genes in subsampled matrix to the genes in the target matrix, and fetch the corresponding expression values.
    The returned subsampled matrix will have the same number and same order of genes as the genes in genes_in_target_mat.

    Parameters:
    - target_gene_by_sample_mat (numpy.ndarray): A genes by samples expression matrix where rows represent genes and columns represent samples.
    - genes_in_target_mat (numpy.ndarray or list): A list/array of gene symbols corresponding to the rows of `target_gene_by_sample_mat`.
    - genes_by_samples_subsampled_mat (numpy.ndarray): A subsampled genes by samples expression matrix.
    - subsampled_gene_symbols (numpy.ndarray or list): A list/array of gene symbols corresponding to the rows of `genes_by_samples_subsampled_mat`.

    Returns:
    - final_matched_matrix (numpy.ndarray): A matrix where rows correspond to genes in `genes_in_target_mat` and columns contain expression values from `genes_by_samples_subsampled_mat`. If a gene in `genes_in_target_mat` is not present in `subsampled_gene_symbols`, the corresponding row will be filled with NaN values.
    - gene_exist_in_subsampled (numpy.ndarray): A boolean array indicating which genes from `genes_in_target_mat` are present in `subsampled_gene_symbols`.

    Notes:
    The function assumes that `genes_in_target_mat` and `subsampled_gene_symbols` are unique lists without repetition.
    """
    gene_by_sample_expr_mat_loged = target_gene_by_sample_mat
    quantified_gene_list = genes_in_target_mat
    # Placeholder for the final matched matrix, filled with NaN
    final_matched_matrix = np.full((gene_by_sample_expr_mat_loged.shape[0], genes_by_samples_subsampled_mat.shape[1]), np.nan)

    
    # For each gene in quantified_gene_list
    for idx, gene in enumerate(quantified_gene_list):
        # Check if gene exists in subsampled_gene_symbols
        if gene in subsampled_gene_symbols:
            # Fetch its index in subsampled_gene_symbols
            gene_idx_in_subsampled = np.where(subsampled_gene_symbols == gene)[0][0]
            
            # Assign the corresponding row from genes_by_samples_subsampled_mat to final_matched_matrix
            final_matched_matrix[idx, :] = genes_by_samples_subsampled_mat[gene_idx_in_subsampled, :]
    
    # Stack gene names and the matrix horizontally
    #final_output = np.hstack((gene_by_sample_expr_mat_loged, final_matched_matrix))
    gene_exist_in_subsampled = np.isin(quantified_gene_list, subsampled_gene_symbols)
    return final_matched_matrix, gene_exist_in_subsampled

def cal_bins_for_target_matrix_old_one_sample_at_a_time(target_gene_by_sample_mat, gene_exist_in_subsampled, matched_subsampled_mat, num_gene_bin):
    """
    Compute binned values for a target genes-by-samples matrix based on a matched subsampled matrix.

    Parameters:
    - target_gene_by_sample_mat (numpy.ndarray): A genes by samples expression matrix where rows represent genes and columns represent samples.
    - gene_exist_in_subsampled (numpy.ndarray): A boolean array indicating which genes from `target_gene_by_sample_mat` are present in the subsampled matrix.
    - matched_subsampled_mat (numpy.ndarray): A matrix where rows correspond to genes in `target_gene_by_sample_mat` and columns contain expression values from a subsampled matrix.
    - num_gene_bin (int): The number of bins to be used for discretization.

    Returns:
    - gene_by_sample_mat_binned (numpy.ndarray): A binned version of `target_gene_by_sample_mat`, with the same shape, where genes present in the subsampled matrix are binned based on their values in `matched_subsampled_mat`, and other genes will be NaN.

    Notes:
    The function uses the `cal_bin_for_each_gene_in_gene_by_sample_mat` function to bin each gene's expression in `target_gene_by_sample_mat` based on its corresponding values in `matched_subsampled_mat`. Genes that are not present in the subsampled matrix will have NaN values in the output.
    """
    gene_by_sample_expr_mat_loged = target_gene_by_sample_mat
    gene_by_sample_mat_binned = np.full(gene_by_sample_expr_mat_loged.shape, np.nan)
    for sample in range(0, gene_by_sample_expr_mat_loged.shape[1]):
        if sample % 10 == 0:
            print(f"processing sample {sample}")
        # Reshape the 1D array to be a 2D column vector
        column_vector = gene_by_sample_expr_mat_loged[:, sample].reshape(-1, 1)
        # Now you can stack them horizontally
        stacked_matrix = np.hstack((column_vector, matched_subsampled_mat))
        bins_of_valid_genes = cal_bin_for_each_gene_in_gene_by_sample_mat(stacked_matrix[gene_exist_in_subsampled], num_gene_bin)
        gene_by_sample_mat_binned[gene_exist_in_subsampled, sample] = bins_of_valid_genes[:, 0]
    return gene_by_sample_mat_binned

def cal_bins_for_target_matrix(target_gene_by_sample_mat, gene_exist_in_subsampled, matched_subsampled_mat, num_gene_bin, chunk_size = 5):
    gene_by_sample_expr_mat_loged = target_gene_by_sample_mat
    gene_by_sample_mat_binned = np.full(gene_by_sample_expr_mat_loged.shape, np.nan)
    
    total_samples = gene_by_sample_expr_mat_loged.shape[1]
    
    for sample_start in range(0, total_samples, chunk_size):
        sample_end = min(sample_start + chunk_size, total_samples)
        if sample_start % 10 == 0:
            print(f"processing samples from {sample_start} to {sample_end-1}")

        sub_matrix = gene_by_sample_expr_mat_loged[:, sample_start:sample_end]
        stacked_matrix = np.hstack((sub_matrix, matched_subsampled_mat))

        # You may need to adjust the bin calculation function if it doesn't support multi-sample matrices.
        bins_of_valid_genes = cal_bin_for_each_gene_in_gene_by_sample_mat(stacked_matrix[gene_exist_in_subsampled], num_gene_bin)

        gene_by_sample_mat_binned[gene_exist_in_subsampled, sample_start:sample_end] = bins_of_valid_genes[:, 0:(sample_end-sample_start)]
    return gene_by_sample_mat_binned



def scale_by_sample_total_count_log_transform(gene_by_sample_expr_mat, output_prefix, min_total_count=1000000, min_expr_genes=2000):

    expression_data = gene_by_sample_expr_mat
    total_counts = np.sum(expression_data, axis=0)
    non_zero_counts = np.count_nonzero(expression_data, axis=0)
    mean_counts = np.mean(expression_data, axis=0)
    median_counts = np.median(expression_data, axis=0)
    max_counts = np.max(expression_data, axis=0)
    valid_samples = (total_counts >= min_total_count) & \
                (non_zero_counts > min_expr_genes)
    expression_data = expression_data[:, valid_samples]
    total_counts_valid = total_counts[valid_samples]
    scaling_factors = 10000000 / total_counts_valid
    normalized_expression = expression_data * scaling_factors[np.newaxis, :]
    log_transformed_expression = np.log10(normalized_expression + 1)
    #np.save(f'{output_prefix}.npy', log_transformed_expression.astype(np.float32))
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
    if output_prefix != None:
        stats_df.to_csv(f'{output_prefix}_stats.tsv', sep='\t', header=True, index=False)
    return log_transformed_expression.astype(np.float32), valid_samples, stats_df





