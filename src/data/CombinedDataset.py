from utils.params import params
params = params()
import os
from train.common_params_funs import BASE_SEED, config
from train.common import train


import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import sys

from data.ReploglePerturbationDataset4Leaper import ReploglePerturbationDataset4Leaper
from utils.checkpoint_utils import find_latest_checkpoint
from collections import OrderedDict
from data.SplitReploglePerturbationDataset import split_dataset


from utils.config_loader import Config
config = Config()
if os.environ.get("RUNNING_MODE") == "debug":
    print("Run in debugging mode!")
    config = Config(config.project_path + "/src/test/config.json")
else:
    print("Run in training mode!")

gene_to_idx_path = config.get("gene2vec_gene_to_idx_json_path")

number_of_special_embeddings = config.get("number_of_special_embeddings")

# params.SAMPLE_NUMBER_FOR_EACH_PERTURBATION = param_finder.find("params.SAMPLE_NUMBER_FOR_EACH_PERTURBATION", 10)

#OUTPUT_ONE_GENE_EVERY = param_finder.find("OUTPUT_ONE_GENE_EVERY", 1)
EMSEMBL2GENE_PATH = config.get("Ensembl_ID_gene_symbol_mapping_file_path")

def adata2dataset(adata, label=None, target_certain_gene_num=None):
    return ReploglePerturbationDataset4Leaper(adata,
                                        gene_to_idx_path=gene_to_idx_path,
                                        emsembl2gene_path=EMSEMBL2GENE_PATH,
                                        n_expr_bins=params.NUM_BINS,
                                        expr_bin_for_fake_additional_gene_output=-1,
                                        label=label,
                                        target_certain_gene_num=target_certain_gene_num
                                        )

class ZipDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.min_length = min(len(dataset1), len(dataset2))

    def __getitem__(self, index):
        item1 = self.dataset1[index]
        item2 = self.dataset2[index]
        return (item1, item2)

    def __len__(self):
        return self.min_length
    
def concat_shuffle_and_extend_datasets(dataset_list, other_dataset_len):
    # Concatenate datasets
    concat_dataset = ConcatDataset(dataset_list)
    # Shuffle the concat_dataset
    indices = torch.randperm(len(concat_dataset)).tolist()
    shuffled_dataset = Subset(concat_dataset, indices)
    # Create a new dataset by repeating the shuffled_dataset n times
    extend_time_n = (other_dataset_len // len(shuffled_dataset))+1
    extended_dataset = ConcatDataset([shuffled_dataset] * extend_time_n)
    return extended_dataset

def get_combined_datasets(dataset_idx, mask_fraction, need_both_pretrain_and_perturb=True):
    
    train_datasets = []
    val_datasets = []
    
    dataset_label_list = ['K562_essential', 'K562_gwps', 'rpe1']
    

    dataset_idx = 0
    tot_perturb_train_len = 0
    tot_perturb_val_len = 0
    perturb_train_dataset_dict = OrderedDict()
    perturb_val_dataset_dict = OrderedDict()
    for dataset_label in dataset_label_list:
    #for dataset_label in ['K562_gwps']:
        dataset_idx += 1
        if params.EXPR_DISCRETIZATION_METHOD == "Direct_quantile":
            file_path_key_in_config=f"fine_tuning_{dataset_label}_dataset_file_path"
        elif params.EXPR_DISCRETIZATION_METHOD == "uniform_bin_count_keep_ones":
            file_path_key_in_config=f"fine_tuning_{dataset_label}_binned_dataset_file_path"
        else:
            print("Unrecognized expr_discretization_method")
        dataset_file = config.get(file_path_key_in_config)
        if params.EXPR_DISCRETIZATION_METHOD == "uniform_bin_count_keep_ones":
            dataset_file = dataset_file.replace("mean_agg.binned", f"mean_agg.{params.GENE_EMB_NAME}.binned")
            print(f"Will use file {dataset_file}")
        elif params.EXPR_DISCRETIZATION_METHOD == "Direct_quantile":
            if params.GENE_EMB_NAME != "gene2vec":
                sys.exit("params.GENE_EMB_NAME != gene2vec and params.EXPR_DISCRETIZATION_METHOD == Direct_quantile")
        if params.USE_AND_KEEP_ZERO_EXPR_GENES == False:
            dataset_file = dataset_file.replace(".mean_agg.", ".mean_agg.without_zero_expr_genes.")
            print("WILL NOT USE ZERO EXPR GENES")
            print(f"use file {dataset_file}")
        train_adata, test_adata = split_dataset(dataset_file = dataset_file, 
                    test_size = 1.0-params.TRAINING_SET_FRACTION, 
                    random_state = BASE_SEED, 
                    baseline_cell_label = "non-targeting")

        train_dataset = adata2dataset(train_adata, dataset_idx, target_certain_gene_num=params.NUM_OF_GENES_SELECTED)
        val_dataset = adata2dataset(test_adata, dataset_idx, target_certain_gene_num=params.NUM_OF_GENES_SELECTED)
        tot_perturb_train_len += len(train_dataset)
        tot_perturb_val_len += len(val_dataset)

        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        perturb_train_dataset_dict[dataset_label] = train_dataset
        perturb_val_dataset_dict[dataset_label] = val_dataset


    return perturb_train_dataset_dict, perturb_val_dataset_dict
