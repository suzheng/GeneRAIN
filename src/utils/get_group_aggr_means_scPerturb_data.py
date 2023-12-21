import argparse
import anndata
import numpy as np
import pandas as pd
import pickle
from data.adata import Adata



def calculate_mean_expression(dataset_file, class_col, output_pickle_file, output_to_tsv, output_tsv_file):
    
    adata = anndata.read_h5ad(dataset_file)
    cell_group_names = adata.obs_names
    gene_names = adata.var_names
    gene_expression_data = pd.DataFrame(adata.X, columns=gene_names)

    classes = adata.obs[class_col]
    # Add class information to the DataFrame
    gene_expression_data['class'] = classes.values.tolist()
    # Group by class and calculate the mean for each gene
    mean_expression_by_class = gene_expression_data.groupby('class').mean()

    mean_expression_by_class_adata = Adata(
        obs_names = mean_expression_by_class.index.values.tolist(),  # Classes as observations
        var_names = mean_expression_by_class.columns.values.tolist(),  # Genes as variables
        expr_matrix = mean_expression_by_class.values  # Mean gene expression values
    )

    # Save the Adata object to a file
    with open(output_pickle_file, 'wb') as f:
        pickle.dump(mean_expression_by_class_adata, f)
    if output_to_tsv:
        # Save mean_expression_by_class DataFrame to a TSV file
        mean_expression_by_class.to_csv(output_tsv_file, sep='\t', index=True, header=True)

        print(f'Data saved to {output_tsv_file}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate mean expression and save to a file.')
    parser.add_argument('--dataset_file', type=str, help='Path to the dataset file.')
    parser.add_argument('--class_col', type=str, default="gene_transcript", help='Column to use for class.')
    parser.add_argument('--output_pickle_file', type=str, default=None, help='Path to save pickle file.')
    parser.add_argument('--output_to_tsv', type=bool, default=True, help='Whether to output to TSV.')
    parser.add_argument('--output_tsv_file', type=str, default=None, help='Path to save TSV file.')
    
    args = parser.parse_args()
    
    if args.output_pickle_file is None:
        args.output_pickle_file = args.dataset_file + ".mean_agg.pickle"
    if args.output_tsv_file is None:
        args.output_tsv_file = args.dataset_file + ".mean_agg.tsv"
    
    calculate_mean_expression(args.dataset_file, args.class_col, args.output_pickle_file, args.output_to_tsv, args.output_tsv_file)
