import pandas as pd

class Adata:
    def __init__(self, obs_names, var_names, expr_matrix):
        self.obs_names = obs_names
        self.var_names = var_names
        self.X = expr_matrix
        self.shape = expr_matrix.shape

    def save_matrix_to_tsv(self, output_tsv_file):
        # Create a DataFrame from the expression matrix
        df = pd.DataFrame(self.X, index=self.obs_names, columns=self.var_names)
        # Save DataFrame to a TSV file
        df.to_csv(output_tsv_file, sep='\t', index=True, header=True)
    