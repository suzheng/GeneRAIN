from utils.params import params
params = params()
import os
import torch
from torch.utils.data import Dataset, ConcatDataset
# param_json_file = '/g/data/yr31/zs2131/tasks/2023/RNA_expr_net/anal/training/exp15/exp15ft_all_genes_on_bin_genes.param_config.json'
# os.environ['PARAM_JSON_FILE'] = param_json_file
from train.common_params_funs import config, get_gene2idx
from train.common import train


from utils.utils import get_device, get_model, get_config, get_gene2idx
from data.CombinedDataset import get_combined_datasets
import numpy as np
import pickle
combined_perturb_train_datasets, combined_perturb_val_datasets = get_combined_datasets(0, 0.0, False)

gene2idx, idx2gene = get_gene2idx()
config = get_config()

res = {}

"""
!!!!
remember you have to use the pred-expr model, as it require gene idx consistent across samples!!!!
and you have to set NUM_OF_GENE_SELECTED to be a number greater than the number of genes in the dataset, otherwise genes will be sorted to select the highest expr genes

!!!!
"""

# dataset_label = 'rpe1' # ['K562_essential', 'K562_gwps', 'rpe1']
for dataset_label in ['K562_essential', 'K562_gwps', 'rpe1']:
    dataset_train = combined_perturb_train_datasets[dataset_label]
    dataset_val = combined_perturb_val_datasets[dataset_label]
    dataset = ConcatDataset([dataset_train, dataset_val])
    
    perturbed_gene_list = []
    output_expr_list = []
    found_pert_gene_idx = {}
    first_gene_index = None
    for i in range(0, len(dataset), dataset.datasets[0].perturb_dataset.sample_number_for_each_perturbation):
        if i % 6000 == 0:
            print(f"processing {i}")
        if first_gene_index is None:
            first_gene_index = dataset[i]['gene_indices']  # Save the value during the first iteration
        else:
            # Check if the current value is the same as the saved value
            assert torch.equal(dataset[i]['gene_indices'], first_gene_index), "Mismatched gene indices in the batch!"
        pert_gene_idx_list = dataset[i]['perturbed_gene_index'].cpu().numpy()
        if pert_gene_idx_list[0] in found_pert_gene_idx:
            continue
        found_pert_gene_idx[pert_gene_idx_list[0]] = 1
        perturbed_gene_list.append(pert_gene_idx_list)
        output_expr_list.append(dataset[i]['output_binned_expr'].cpu().numpy())
    pert_real_conseq = np.array(output_expr_list)
    pert_real_conseq_gene_idx = np.array(perturbed_gene_list).squeeze(-1)
    genes_in_pert_real_conseq = [idx2gene[idx] for idx in pert_real_conseq_gene_idx]
    gene2idx_for_pert_real_conseq = {genes_in_pert_real_conseq[i]: i for i in range(len(genes_in_pert_real_conseq))}
    res[dataset_label] = {"pert_gene_by_expr_gene_mat": pert_real_conseq,
                          "gene2idx_in_the_mat": gene2idx_for_pert_real_conseq
                         }
with open(f"{config.proj_path}/data/external/scPerturb/data/pert_gene_by_binned_expr_gene_mat_{params.GENE_EMB_NAME}.pkl", 'wb') as file:
    pickle.dump(res, file)