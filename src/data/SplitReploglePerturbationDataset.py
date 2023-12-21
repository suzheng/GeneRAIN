import anndata
import numpy as np
from utils.config_loader import Config
from sklearn.model_selection import train_test_split
import pickle
from data.adata import Adata

config = Config()


#this function split the dataset into two Adata objects, which can be used to 
#create traing and setting ReploglePerturbationDataset without any overlapping cells
def split_dataset(dataset_file, test_size, random_state, baseline_cell_label = "non-targeting"):
    #adata = anndata.read_h5ad(dataset_file)
    # Load the Adata object from a file
    
    with open(dataset_file, 'rb') as f:
        adata = pickle.load(f)
    #print(dataset_file)
    cell_group_names = adata.obs_names
    gene_names = adata.var_names
    expr_matrix = adata.X

    # Create a binary list indicating whether "non-targeting" is contained in each cell group name
    stratify_labels = [baseline_cell_label in name for name in cell_group_names]

    # Stratified Split the data

    train_cell_groups, test_cell_groups, train_expr_matrix, test_expr_matrix = train_test_split(
        cell_group_names, expr_matrix, test_size=test_size, random_state=random_state, stratify=stratify_labels
    )
    

    # Create Adata objects for training and testing
    train_adata = Adata(train_cell_groups, gene_names, train_expr_matrix)
    test_adata = Adata(test_cell_groups, gene_names, test_expr_matrix)
    return train_adata, test_adata
