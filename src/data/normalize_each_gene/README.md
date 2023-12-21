## About use_and_keep_zero_expr_genes
Whether use_and_keep_zero_expr_genes is True or not, the uniform bining on each genes will always make the zero expression genes have bin values of 1, and the binning operating on the normalized log_transformed raw expression data, no z-score was performed. So the use_and_keep_zero_expr_genes value (being true or false) has no effect on the binning values.
The effect it has is on the selection of genes using 'use_and_keep_zero_expr_genes' by script 5_combine_genes_filter_genes_split_into_sample_chunks_for_bins.py. But it is recommended to use use_and_keep_zero_expr_genes==True, as it effectively filter out genes that have zero expression in almost all samples.
After binning, zero expression genes will always have bin value of 1

    archs_dataset = ARCHSDataset(h5_file_path, gene_stat_uniq_bool_file=output_file_prefix + ".gene_stat_filt_on_z_dup.tsv")
    idx_genes_of_interest = archs_dataset.in_gene2vec_nondup_bool
    #output_file_prefix = "/path/to/your/files/"
    merged_matrix = merge_sample_chunks(output_file_prefix, total_chunks, chunk_idx, idx_genes_of_interest, num_bins)


use_and_keep_zero_expr_genes have effect on z-scores. 
if use_and_keep_zero_expr_genes == True, z-score will be calculated on all expression, regardless it has zero expression or not. Zero expression genes will have small negative values.
if use_and_keep_zero_expr_genes == False, z-score will be calculated on using the means and std of non-zero expression genes. Then the zero expression genes will be assigned as z-score value of zero, and that zero value can be used to get the bool array of zero_expression by  discretize_expression_zscores(expression_vector, n_bins=100) in src/data/discretize_expression.py

