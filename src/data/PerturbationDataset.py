from utils.params import params
params = params()
import anndata
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.config_loader import Config
from utils.Ensembl_ID_gene_symbol_mapping_reader import read_ensembl_to_gene_mapping
from utils.json_utils import JsonUtils
from data.discretize_expression import discretize_expression_zscores, discretize_expression_uniform
from utils.string_tensor import string_to_tensor
from data.discretize_expression import uniform_bin_count_keep_ones
from data.data_utils import get_top_genes, find_indices_of_elements_in_an_array
from train.common_params_funs import config
from train.common import train

config = Config()
ju = JsonUtils()
import sys

# Your existing code and functions
def random_selection(input_list, num_elements, seed_value=None):
    if seed_value is not None:
        random.seed(seed_value)
    if num_elements >= len(input_list):
        return input_list
    else:
        return random.sample(input_list, num_elements)
# None of dataset will have special tokens, all special tokens will be added outside of GeneExprTransformer Module, inside the wrapper modules.
# In in GA paper analysis, sample_number_for_each_perturbation was set to 10 directly
class PerturbationDataset(Dataset):
    """
    samples_by_genes_expr_mat, the input matrix
    quantified_gene_list, should have a same length as the column num of samples_by_genes_expr_mat
    perturbed_gene_list, should have a same length as the row num of samples_by_genes_expr_mat
    """
    def __init__(
        self,
        samples_by_genes_expr_mat,
        quantified_gene_list,
        perturbed_gene_list,
        gene_to_idx_dict,
        new_expr_bin_val,
        sample_number_for_each_perturbation=params.SAMPLE_NUMBER_FOR_EACH_PERTURBATION,
        n_expr_bins=params.NUM_BINS,
        output_one_gene_every=1,
        expr_bin_for_fake_additional_gene_output=-1,
        dataset_label=None,
        target_certain_gene_num=params.NUM_OF_GENES_SELECTED,
        number_of_special_embeddings=params.NUMBER_OF_SPECIAL_TOKEN_IN_DATASET,
        only_use_postive_zscores_in_training= params.ONLY_USE_POSITIVE_ZSCORES_IN_TRAINING,
        perturbed_gene_always_in_input_expr=params.PERTURBED_GENE_ALWAYS_IN_INPUT_EXPR_IN_PERTURB_DATASET,
        expr_discretization_method=params.EXPR_DISCRETIZATION_METHOD,
        sort_return_expr_numerically=(params.TRANSFORMER_MODEL_NAME == "GPT" or params.TRANSFORMER_MODEL_NAME == "Bert_pred_tokens")
    ):
        self.sample_number_for_each_perturbation = sample_number_for_each_perturbation
        #self.config = config
        #self.file_path_key_in_config = file_path_key_in_config
        self.n_expr_bins = n_expr_bins
        self.new_expr_bin_val = new_expr_bin_val
        self.perturbed_gene_list_for_all_samples = np.array(perturbed_gene_list, dtype=object)
        if not isinstance(new_expr_bin_val, (list, tuple)):
            self.new_expr_bin_val = np.array([new_expr_bin_val] * len(self.perturbed_gene_list_for_all_samples))
        self.expr_bin_for_fake_additional_gene_output = expr_bin_for_fake_additional_gene_output
        self.output_one_gene_every = output_one_gene_every
        self.gene_to_idx_dict = gene_to_idx_dict
        self.idx_to_gene_dict = {v:k for k,v in gene_to_idx_dict.items()}
        self.samples_by_genes_expr_mat = samples_by_genes_expr_mat
        self.inf_count = 0
        self.neg_inf_count = 0
        self.nan_count = 0
        self.dataset_label = [dataset_label]
        self.target_certain_len = target_certain_gene_num
        self.number_of_special_embeddings = number_of_special_embeddings
        self.only_use_postive_zscores_in_training = only_use_postive_zscores_in_training
        self.perturbed_gene_always_in_input_expr = perturbed_gene_always_in_input_expr
        self.expr_discretization_method = expr_discretization_method
        self.sort_return_expr_numerically = sort_return_expr_numerically
        # the key here are the strings for matching the string at the position of gene_symbol in baseline cell groups
        baseline_keys = ["control"]
        possible_expr_discretization_methods = ["Direct_quantile", "uniform_bin_count_keep_ones"]
        if self.expr_discretization_method not in possible_expr_discretization_methods:
            print(f"expr_discretization_method has to be one of {possible_expr_discretization_methods}, but {self.expr_discretization_method} was found!")
            sys.exit()
        def get_indices_of_two_groups_of_samples_in_expr_mat(perturbed_gene_list_for_all_samples, baseline_keys, gene_to_idx_dict):
            indices_of_perturb_samples_in_expr_mat = []
            indices_of_baseline_samples_in_expr_mat = []

            for i, perturbed_genes_of_a_single_sample in enumerate(perturbed_gene_list_for_all_samples):

                if not isinstance(perturbed_genes_of_a_single_sample, (list, np.ndarray)):
                    if perturbed_genes_of_a_single_sample in baseline_keys:
                        indices_of_baseline_samples_in_expr_mat.append(i)
                    elif perturbed_genes_of_a_single_sample in gene_to_idx_dict:
                        indices_of_perturb_samples_in_expr_mat.append(i)
                else:
                    # indices_baseline_in_expr_mat_for_a_single_sample = []
                    for perturbed_gene_of_a_single_sample in perturbed_genes_of_a_single_sample:
                        if perturbed_gene_of_a_single_sample in baseline_keys:
                            indices_of_baseline_samples_in_expr_mat.append(i)
                            # as a sample can only be neither baseline or perturbed, if baseline is found, won't go to checking if is perturb 
                            break
                        elif perturbed_gene_of_a_single_sample in gene_to_idx_dict:
                            indices_of_perturb_samples_in_expr_mat.append(i)
                            break

            # Convert the lists to numpy arrays before returning
            return indices_of_baseline_samples_in_expr_mat, indices_of_perturb_samples_in_expr_mat

        ## PROCESS SAMPLES
        # Assuming perturbed_gene_list_for_all_samples, baseline_keys, and gene_to_idx_dict are defined
        indices_of_baseline_samples_in_expr_mat, indices_of_perturb_samples_in_expr_mat = get_indices_of_two_groups_of_samples_in_expr_mat(self.perturbed_gene_list_for_all_samples, baseline_keys, self.gene_to_idx_dict)


        ## PROCESS GENES
        #only select the genes that have mapped gene symbols and have embedding index in gene_to_dix
        #the indices here are the indices of the overlapping genes in orginal h5ad file
        self.idx_overlapping_genes_in_expr_mat = []
        #symbols of genes that have mapped gene symbols and have embedding index in gene_to_dix
        self.overlapping_gene_symbols = []
        #the indices here are the indices of the overlapping genes in geneID embedding
        self.indices_overlapping_genes_in_gene2vec = []
        found = 0
        for i, gene_symbol in enumerate(quantified_gene_list):
            #still works if gene is gene symbol but not ensembl ID
            # gene_symbol = self.emsembl2gene.get(gene, gene)
            if gene_symbol in self.gene_to_idx_dict:
                self.overlapping_gene_symbols.append(gene_symbol)
                self.idx_overlapping_genes_in_expr_mat.append(i)
                self.indices_overlapping_genes_in_gene2vec.append(self.gene_to_idx_dict[gene_symbol])
        self.indices_overlapping_genes_in_gene2vec = np.array(self.indices_overlapping_genes_in_gene2vec)
        if sample_number_for_each_perturbation > len(indices_of_baseline_samples_in_expr_mat):
            sample_number_for_each_perturbation = len(indices_of_baseline_samples_in_expr_mat)
        #self.dataset_length = len(indices_of_baseline_samples_in_expr_mat) + sample_number_for_each_perturbation * len(indices_of_perturb_samples_in_expr_mat)

        self.dataset_metadata = {}
        i = 0

        # a sample may have multiple perturbed genes
        for idx_perturbation_in_expr_mat in indices_of_perturb_samples_in_expr_mat:

            baseline_indices_in_expr_mat_for_this_perturb = random_selection(indices_of_baseline_samples_in_expr_mat, sample_number_for_each_perturbation)
            for baseline_idx_in_expr_mat_for_this_perturb in baseline_indices_in_expr_mat_for_this_perturb:
                self.dataset_metadata[i] = {"idx_perturbation_in_expr_mat": idx_perturbation_in_expr_mat,
                                        "baseline_idx_in_expr_mat_for_this_perturb": baseline_idx_in_expr_mat_for_this_perturb,
                                        "perturbed_gene": self.perturbed_gene_list_for_all_samples[idx_perturbation_in_expr_mat],
                                        "new_expr_bin": self.new_expr_bin_val[idx_perturbation_in_expr_mat]
                                        }
                i += 1
        self.dataset_length = i


    
    def __len__(self):
        return self.dataset_length
    
    ## to_change if other data set doesn't use inf, -inf, and NaN
    def sanitize_input_raw_expr(self, input_raw_expr):
        # Find indices of inf, -inf, and NaN values
        inf_indices = np.isinf(input_raw_expr)
        neg_inf_indices = np.isneginf(input_raw_expr)
        nan_indices = np.isnan(input_raw_expr)

        # Temporarily replace inf and -inf with NaN to find the max and min finite values
        temp_input = input_raw_expr.copy()
        temp_input[np.logical_or(inf_indices, neg_inf_indices)] = np.nan
        max_value = np.nanmax(temp_input)
        min_value = np.nanmin(temp_input)

        # Replace inf, -inf, and NaN values
        input_raw_expr[inf_indices] = max_value
        input_raw_expr[neg_inf_indices] = min_value

        # Update the counters
        self.inf_count += np.sum(inf_indices)
        self.neg_inf_count += np.sum(neg_inf_indices)
        self.nan_count += np.sum(nan_indices)

        # Replace NaN values
        input_raw_expr[nan_indices] = np.random.uniform(min_value, max_value, size=np.sum(nan_indices))

        return input_raw_expr

    def sort_result_based_on_expr(self, result, expression_vector_float32):
        if not torch.is_tensor(expression_vector_float32):
            expression_vector_float32 = torch.tensor(expression_vector_float32)
        sorted_indices = torch.argsort(expression_vector_float32, descending=True)
        for key, value in result.items():
            if value.numel() == expression_vector_float32.numel():
                result[key] = value[sorted_indices].clone().detach()
        return result

    def prepend_special_tokens(self, result, number_of_special_embeddings):
        if number_of_special_embeddings == 0:
            return result
        # Prepend indices for gene_indices
        special_tokens_indices = torch.arange(number_of_special_embeddings, dtype=result['gene_indices'].dtype)
        result['gene_indices'] = torch.cat([special_tokens_indices, result['gene_indices']])
        if 'gene_indices_perturbed' in result:
            result['gene_indices_perturbed'] = torch.cat([special_tokens_indices, result['gene_indices_perturbed']])
        # Prepend zeros for input_binned_expr and output_binned_expr
        zeros_to_prepend = torch.zeros(number_of_special_embeddings, dtype=result['input_binned_expr'].dtype)
        result['input_binned_expr'] = torch.cat([zeros_to_prepend, result['input_binned_expr']])
        result['output_binned_expr'] = torch.cat([zeros_to_prepend, result['output_binned_expr']])
        result['input_binned_expr_pert_gene_changed'] = torch.cat([zeros_to_prepend, result['input_binned_expr_pert_gene_changed']])
        
        # Prepend False for zero_expression_genes
        false_to_prepend = torch.zeros(number_of_special_embeddings, dtype=torch.bool)
        result['zero_expression_genes'] = torch.cat([false_to_prepend, result['zero_expression_genes']])
        
        return result
        # input numpy array
    # output numpy arrays of top genes, and the numeric indices

    def get_top_binned_expr(self, idx_in_expr_mat, top_gene_num, top_abs_indices_in_overlapping_genes=None, index_will_always_be_included_in_finding_top_genes=None):
        
        #all the expr here are for overlapping genes (overlap with gene2vec) only
        raw_expr = self.samples_by_genes_expr_mat[idx_in_expr_mat, self.idx_overlapping_genes_in_expr_mat]
        raw_expr = self.sanitize_input_raw_expr(raw_expr)
        
        if top_abs_indices_in_overlapping_genes is None:
            top_raw_expr, top_abs_indices_in_overlapping_genes = get_top_genes(raw_expr, 
                                                                               top_gene_num, 
                                                                               self.only_use_postive_zscores_in_training, 
                                                                               index_will_always_be_included=index_will_always_be_included_in_finding_top_genes)
        else:
            top_raw_expr = raw_expr[top_abs_indices_in_overlapping_genes]
            
        if self.expr_discretization_method == "uniform_bin_count_keep_ones":
            binned_expr, zero_expression_genes = uniform_bin_count_keep_ones(top_raw_expr, self.n_expr_bins)
        elif self.expr_discretization_method == "Direct_quantile":
            binned_expr, zero_expression_genes = discretize_expression_zscores(top_raw_expr, self.n_expr_bins)
            
        indices_top_genes_in_gene2vec_np_array = self.indices_overlapping_genes_in_gene2vec[top_abs_indices_in_overlapping_genes]
        
        return binned_expr, zero_expression_genes, indices_top_genes_in_gene2vec_np_array, top_abs_indices_in_overlapping_genes, top_raw_expr
    

    def get_expr_of_top_genes_pert_gene_changed(self, indices_top_genes_in_gene2vec_np_array, perturbed_gene_index_in_gene2vec_no_duplicate, expr_of_top_genes, perturbed_gene_expr_no_duplicate):
        """
        This method changes the expression values of each perturbed gene

        Parameters:
        indices_top_genes_in_gene2vec_np_array (np.ndarray): An array of indices representing the top genes in gene2vec.
        perturbed_gene_index_in_gene2vec_no_duplicate (list): A list of unique indices representing the perturbed genes in gene2vec.
        expr_of_top_genes (np.ndarray or list): An array or list containing the expression values of the top genes, expr hasn't been discretized
        perturbed_gene_expr_no_duplicate (np.ndarray or list): An array or list containing the target expression values of the perturbed genes.

        Returns:
        np.ndarray or list: An array or list containing the adjusted expression values of the top genes, with the expression of perturbed genes modified according to their perturbation direction.

        Note:
        The method first finds the indices of perturbed genes within the top genes. Then, it creates a copy of perturbed gene expressions and adjusts the expression values if they equal 'self.n_expr_bins'. Finally, it creates a copy of the expression values of top genes and modifies the expression of perturbed genes in the list, returning the adjusted expression values.
        """
        idx_pert_genes_in_top_genes = find_indices_of_elements_in_an_array(indices_top_genes_in_gene2vec_np_array, perturbed_gene_index_in_gene2vec_no_duplicate)
        perturbed_gene_expr_direction = perturbed_gene_expr_no_duplicate.copy()
        for i in range(len(perturbed_gene_expr_direction)):
            if perturbed_gene_expr_direction[i] == self.n_expr_bins:
                perturbed_gene_expr_direction[i] = max(expr_of_top_genes) + 1
            elif self.expr_discretization_method == "Direct_quantile" and (perturbed_gene_expr_direction[i] == 0 or perturbed_gene_expr_direction[i] == 1):
                perturbed_gene_expr_direction[i] = min(expr_of_top_genes)

        expr_of_top_genes_pert_gene_changed = expr_of_top_genes.copy()
        if len(idx_pert_genes_in_top_genes) >0:
            #print(f"idx_pert_genes_in_top_genes {idx_pert_genes_in_top_genes}, perturbed_gene_expr_direction {perturbed_gene_expr_direction}")
            expr_of_top_genes_pert_gene_changed[idx_pert_genes_in_top_genes] = perturbed_gene_expr_direction
        return expr_of_top_genes_pert_gene_changed
    
    def __getitem__(self, idx):
        #print(f"idx: {idx}")
        one_meta_data = self.dataset_metadata[idx]
        # Replicate perturbed genes by 100 times for perturbation cell lines
        perturbed_gene = one_meta_data["perturbed_gene"]
        # Check if perturbed_gene is a numpy array or a list
        if isinstance(perturbed_gene, (np.ndarray, list)):
            # If it is an array or list, iterate over its elements and get the corresponding values from the dictionary
            perturbed_gene_index_in_gene2vec = [self.gene_to_idx_dict[gene] for gene in perturbed_gene]
            perturbed_gene_index_in_gene2vec_no_duplicate = list(set(perturbed_gene_index_in_gene2vec))
        else:
            # If it is a single gene, get the corresponding value from the dictionary
            perturbed_gene_index_in_gene2vec = self.gene_to_idx_dict[perturbed_gene]
            perturbed_gene_index_in_gene2vec_no_duplicate = [perturbed_gene_index_in_gene2vec]
        
        # Adjust the expression of the perturbed gene
        expr_of_perturbed_gene = one_meta_data["new_expr_bin"]
        perturbed_gene_expr_no_duplicate = [expr_of_perturbed_gene]
        
        perburbed_gene_index_in_overlapping_genes = find_indices_of_elements_in_an_array(self.indices_overlapping_genes_in_gene2vec, perturbed_gene_index_in_gene2vec_no_duplicate)
        
        index_will_always_be_included_in_finding_top_genes = None
        if self.perturbed_gene_always_in_input_expr == True:
            index_will_always_be_included_in_finding_top_genes = perburbed_gene_index_in_overlapping_genes
        #one_meta_data["baseline_idx_in_expr_mat_for_this_perturb"] = 137
        input_binned_expr, input_zero_expression_genes, indices_top_genes_in_gene2vec_np_array, top_abs_indices_in_overlapping_genes, top_raw_input_expr = self.get_top_binned_expr(one_meta_data["baseline_idx_in_expr_mat_for_this_perturb"], 
                                                                                                                                                                                    self.target_certain_len, 
                                                                                                                                                                                    top_abs_indices_in_overlapping_genes=None,
                                                                                                                                                                                    index_will_always_be_included_in_finding_top_genes=index_will_always_be_included_in_finding_top_genes
                                                                                                                                                                                    )
        
        

        output_binned_expr, output_zero_expression_genes, _, _, top_raw_output_expr = self.get_top_binned_expr(one_meta_data["idx_perturbation_in_expr_mat"], self.target_certain_len, top_abs_indices_in_overlapping_genes=top_abs_indices_in_overlapping_genes)


        expr_of_top_genes_pert_gene_changed = self.get_expr_of_top_genes_pert_gene_changed(indices_top_genes_in_gene2vec_np_array, perturbed_gene_index_in_gene2vec_no_duplicate, top_raw_input_expr, perturbed_gene_expr_no_duplicate)
        if self.expr_discretization_method == "uniform_bin_count_keep_ones":
            input_binned_expr_pert_gene_changed, _ = uniform_bin_count_keep_ones(expr_of_top_genes_pert_gene_changed, self.n_expr_bins)
        elif self.expr_discretization_method == "Direct_quantile":
            input_binned_expr_pert_gene_changed, _ = discretize_expression_zscores(expr_of_top_genes_pert_gene_changed, self.n_expr_bins)
        


        #ret_gene_symbols = [self.idx_to_gene_dict[idx] for idx in indices_top_genes_in_gene2vec_np_array]
        ret_gene_indices = torch.tensor(indices_top_genes_in_gene2vec_np_array)

        # "dataset_label": self.label,
        # "gene_symbols": ret_gene_symbols,
        # "top_raw_input_expr": torch.tensor(top_raw_input_expr, dtype=torch.float32),
        # "top_raw_output_expr": torch.tensor(top_raw_output_expr, dtype=torch.float32),
        result = {
            "dataset_label": torch.tensor(self.dataset_label),
            "baseline_idx_in_expr_mat_for_this_perturb":torch.tensor([one_meta_data["baseline_idx_in_expr_mat_for_this_perturb"]], dtype=torch.int32),
            "idx_perturbation_in_expr_mat":torch.tensor([one_meta_data["idx_perturbation_in_expr_mat"]], dtype=torch.int32),
            "gene_indices": ret_gene_indices,
            "input_binned_expr": torch.tensor(input_binned_expr, dtype=torch.int32),
            "output_binned_expr": torch.tensor(output_binned_expr, dtype=torch.int32),
            "input_binned_expr_pert_gene_changed": torch.tensor(input_binned_expr_pert_gene_changed, dtype=torch.int32),
            "zero_expression_genes": torch.zeros_like(ret_gene_indices, dtype=torch.bool),
            "perturbed_gene_index": torch.tensor(perturbed_gene_index_in_gene2vec_no_duplicate),
            "perturbed_gene_expr": torch.tensor(perturbed_gene_expr_no_duplicate)
        }
        # print(f"top_raw_input_expr: {top_raw_input_expr}")
        # print(f"expr_of_top_genes_pert_gene_changed: {expr_of_top_genes_pert_gene_changed}")
        # print(result)
        if self.sort_return_expr_numerically:
            perturb_gene_indices_result = {"gene_indices_perturbed": result["gene_indices"].clone().detach()}
            perturb_gene_indices_result_sorted = self.sort_result_based_on_expr(perturb_gene_indices_result, expr_of_top_genes_pert_gene_changed)
            result = self.sort_result_based_on_expr(result, top_raw_input_expr)
            result["gene_indices_perturbed"] = perturb_gene_indices_result_sorted["gene_indices_perturbed"]
        # print(result)
        result = self.prepend_special_tokens(result, self.number_of_special_embeddings)

        # return self.add_to_certain_len(result, self.target_certain_len)
        return result
        
  
    def __del__(self):
        #self.adata.file.close()
        print("Number of records with inf values:", self.inf_count)
        print("Number of records with -inf values:", self.neg_inf_count)
        print("Number of records with NaN values:", self.nan_count)


