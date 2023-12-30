from utils.params import params
params = params()
import random
import torch
import numpy as np
import pandas as pd
from utils.config_loader import Config
from utils.Ensembl_ID_gene_symbol_mapping_reader import read_ensembl_to_gene_mapping
from utils.json_utils import JsonUtils
from train.common_params_funs import config
from train.common import train

config = Config()
ju = JsonUtils()

def normalize_expr_mat(gene_stat_file, gene_by_sample_expr_mat, quantified_gene_list, output_prefix="./stats_df", min_mean_val_for_zscore=0.1, min_total_count=-1, use_and_keep_zero_expr_genes=params.USE_AND_KEEP_ZERO_EXPR_GENES):
    """
    Process gene data to produce zscores.

    Parameters:
    - gene_stat_file: The file containing gene statistics.
    - gene_by_sample_expr_mat: Gene expression matrix.
    - quantified_gene_list: List of genes of interest.
    - min_mean_val_for_zscore: Minimum mean value for zscore calculation.
    - output_prefix: Prefix for output data.
    - min_total_count: Minimum total count.

    Returns:
    - gene_by_sample_zscores: Zscores for each gene sample.
    - valid_samples, List of booleans indicating if the samples are qualified
    - stats_df
    """
    gene_stat_df = get_gene_stat_df(gene_stat_file, min_mean_val_for_zscore)
    
    gene_by_sample_expr_mat_loged, valid_samples, stats_df = scale_by_sample_total_count_log_transform(gene_by_sample_expr_mat, output_prefix=output_prefix, min_total_count=min_total_count)
    
    gene_mean, gene_std = get_mean_std_for_genes_of_interest(quantified_gene_list, gene_stat_df)
    
    gene_by_sample_zscores = get_zscores(gene_by_sample_expr_mat_loged, gene_mean, gene_std, min_mean_val_for_zscore, use_and_keep_zero_expr_genes=use_and_keep_zero_expr_genes)
    
    return gene_by_sample_zscores, valid_samples, stats_df


def get_gene_stat_df(gene_stat_file, min_mean_val_for_zscore):
    
    # Concatenate all the dataframes in the list into a single dataframe
    gene_stat_df = pd.read_csv(gene_stat_file, sep="\t")
    gene_stat_df['gene_symbol'] = gene_stat_df['gene_symbol'].str.replace("b'", "").str.replace("'", "")
    gene_stat_df = gene_stat_df[(gene_stat_df['max_mean_nondup'] == True) & (gene_stat_df['gene_mean'] > min_mean_val_for_zscore)]
    gene_stat_df.set_index('gene_symbol', inplace=True)
    return gene_stat_df


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
    stats_df.to_csv(f'{output_prefix}_stats.tsv', sep='\t', header=True, index=False)
    return log_transformed_expression.astype(np.float32), valid_samples, stats_df


def get_mean_std_for_genes_of_interest(genes_of_interest, gene_stat_df):
    # Create a DataFrame with gene names for merging
    gene_names_df = pd.DataFrame(genes_of_interest, columns=['gene_symbol'])
    # Merge with gene_stat_df
    merged_df = pd.merge(gene_names_df, gene_stat_df, left_on='gene_symbol', right_index=True, how='left')
    #print(merged_df.iloc[0:3, :])
    print("Total number of values per column:\n", merged_df.iloc[:, 1:].count())
    print("Number of missing values per column:\n", merged_df.iloc[:, 1:].isna().sum())
    # Fill missing values with 0s (or any other value deemed appropriate)
    merged_df.fillna(0, inplace=True)
    gene_mean = merged_df.iloc[:, 1].values
    gene_std = merged_df.iloc[:, 2].values
    #merged_df.to_csv("merged_df.tsv", sep="\t", index=False)
    return gene_mean, gene_std


def get_zscores(gene_by_sample_expr_mat_loged, gene_mean, gene_std, min_mean_val_for_zscore, use_and_keep_zero_expr_genes):
    if use_and_keep_zero_expr_genes:
        masked_expression = np.ma.masked_equal(gene_by_sample_expr_mat_loged, -1)
    else:
        masked_expression = np.ma.masked_equal(gene_by_sample_expr_mat_loged, 0)
    z_score_expression = np.zeros_like(gene_by_sample_expr_mat_loged)
    zero_expr = masked_expression.mask
    # Calculate the z-score for each gene
    valid_genes = np.where(gene_mean >= min_mean_val_for_zscore)[0]
    for gene in valid_genes:
        z_score_expression[gene, :] = (gene_by_sample_expr_mat_loged[gene, :] - gene_mean[gene]) / gene_std[gene]
    # Replace any NaNs or Infs that might result from division by zero
    value_to_replace_nan = 0
    z_score_expression = np.nan_to_num(z_score_expression, nan=value_to_replace_nan, posinf=value_to_replace_nan, neginf=value_to_replace_nan)
    z_score_expression[zero_expr] = 0
    return z_score_expression



