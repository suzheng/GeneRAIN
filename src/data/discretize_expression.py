import numpy as np
import scipy.stats as stats

# use normal distribution
def discretize_expression_normal_distr(expression_vector, n_bins=100, std_dev=1.1):
    # Determine non-zero values
    nonzero_indices = np.nonzero(expression_vector)[0]
    expression_vector_nonzero = expression_vector[nonzero_indices]

    # Rank-based inverse normal transformation
    ranks = stats.rankdata(expression_vector_nonzero)
    transformed_std = stats.norm(loc=0, scale=1).ppf(ranks / (len(ranks) + 1))
    transformed = stats.norm(loc=0, scale=std_dev).ppf(ranks / (len(ranks) + 1))

    # Define bin edges from minimum to maximum
    bin_edges = np.linspace(np.min(transformed_std), np.max(transformed_std), n_bins-1)

    # Discretize transformed values
    discretized_expression_nonzero = np.digitize(transformed, bin_edges, right=True)

    # Add 1 because np.digitize returns 0 for elements that are less than the first bin edge
    discretized_expression_nonzero += 1

    # Create an array to store the discretized values, initialized with zeros
    discretized_expression = np.zeros_like(expression_vector, dtype=int)

    # Fill the positions of non-zero values with the discretized expression values
    discretized_expression[nonzero_indices] = discretized_expression_nonzero

    # Create a boolean array indicating non-zero gene expressions
    zero_expression_genes = np.ones_like(expression_vector, dtype=bool)
    zero_expression_genes[nonzero_indices] = False

    return discretized_expression, zero_expression_genes

# We have to keep the 1s, as zero expression genes will always have bin value of 1, regardless use_and_keep_zero_expr_genes being true or not
def uniform_bin_count_keep_ones(expression_vector, n_bins=100):
    # Create a copy to not modify the original data
    discretized_vector = expression_vector.copy()

    # Only consider values greater than 1 for discretization
    values_to_discretize = expression_vector > 1

    # Extract those values and discretize them
    extracted_values = expression_vector[values_to_discretize]
    quantiles = np.linspace(0, 1, n_bins+1)
    bin_edges = np.quantile(extracted_values, quantiles)
    discretized_values = np.digitize(extracted_values, bin_edges, right=True)
    discretized_values[discretized_values<1] = 1

    # Place the discretized values back into the copied vector
    discretized_vector[values_to_discretize] = discretized_values
    #print(expression_vector)
    zero_expression_genes = (expression_vector <= 1)
    return discretized_vector, zero_expression_genes


# in z-scores were set to 0 for ZERO expression genes in {output_file_prefix}_final_chunk_{chunk_idx}.npy, if use_and_keep_zero_expr_genes == False
# if use_and_keep_zero_expr_genes == True, zero expression genes will have small (negative) values, in this case, we can't provide the bool list of zero expression genes
def discretize_expression_zscores(expression_vector, n_bins=100):
    quantiles = np.linspace(0, 1, n_bins)
    bin_edges = np.quantile(expression_vector, quantiles)
    discretized_expression = np.digitize(expression_vector, bin_edges, right=True)
    return discretized_expression + 1, expression_vector == 0

# def discretize_expression(expression_vector, n_bins=100):
#     return discretize_expression_zscores(expression_vector, n_bins)

def discretize_expression_uniform(ori_expression_vector, n_bins=100):
    # Separate zero-expression genes from the rest
    zero_expression_genes = ori_expression_vector == 0
    non_zero_expression_genes_indices = np.where(~zero_expression_genes)[0]

    non_zero_expression_genes = ori_expression_vector[non_zero_expression_genes_indices]
    non_zero_expression_genes_sorted_indices = np.argsort(non_zero_expression_genes)
    num_non_zero_genes = len(non_zero_expression_genes)
    genes_per_bin = num_non_zero_genes // n_bins

    # Create an empty array for discretized expression
    # minimum expression has bin value of 1, but not 0, 0 is reserved for random embedding for masking
    discretized_expression = np.ones_like(ori_expression_vector)

    # Assign bin indices based on the sorted position
    for i in range(n_bins):
        start = i * genes_per_bin
        if i == n_bins - 1:  # The last bin may contain more genes if num_non_zero_genes is not divisible by n_bins
            end = num_non_zero_genes
        else:
            end = start + genes_per_bin
        bin_indices = non_zero_expression_genes_sorted_indices[start:end]
        discretized_expression[non_zero_expression_genes_indices[bin_indices]] = i + 1  # +1 so that the first bin is 1, not 0

    return discretized_expression, zero_expression_genes
