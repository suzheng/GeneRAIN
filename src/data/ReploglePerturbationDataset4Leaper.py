import anndata
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.config_loader import Config
from utils.Ensembl_ID_gene_symbol_mapping_reader import read_ensembl_to_gene_mapping
from utils.json_utils import JsonUtils
from utils.string_tensor import string_to_tensor
from data.PerturbationDataset import PerturbationDataset
from data.adata import Adata
config = Config()
ju = JsonUtils()
from utils.utils import get_gene2idx



class ReploglePerturbationDataset4Leaper(Dataset):
    def __init__(
        self,
        adata,
        gene_to_idx_path=None,
        emsembl2gene_path=None,
        n_expr_bins=None,
        output_one_gene_every=1,
        new_expr_bin_val=1,
        expr_bin_for_fake_additional_gene_output=-1,
        label=None,
        target_certain_gene_num=None
    ):
        
        emsembl2gene = (
            read_ensembl_to_gene_mapping(emsembl2gene_path) if emsembl2gene_path else None
        )
        #gene_to_idx_dict = ju.load_data_from_file(gene_to_idx_path)
        gene_to_idx_dict, _ = get_gene2idx()
        cell_group_names = adata.obs_names
        quantified_gene_list = adata.var_names
        samples_by_genes_expr_mat = adata.X
        baseline_keys = ["non-targeting"]
                #including the baseline and perturbed gene symbols, has the same length as original h5ad matrix
        perturbed_gene_list = []
        #indices all starts from zero
        for i, cell_group_name in enumerate(cell_group_names):
            perturbed_gene = cell_group_name.split("_")[1]
            
            if perturbed_gene in baseline_keys:
                perturbed_gene = "control"

            perturbed_gene_list.append(perturbed_gene)
        self.perturb_dataset = PerturbationDataset(samples_by_genes_expr_mat, 
                                                            quantified_gene_list,
                                                            perturbed_gene_list,
                                                            gene_to_idx_dict,
                                                            new_expr_bin_val,
                                                            dataset_label=label,
                                                            target_certain_gene_num = target_certain_gene_num
                                                            )
    def __len__(self):
        return self.perturb_dataset.__len__()

    def __getitem__(self, index):
        return self.perturb_dataset.__getitem__(index)
        



