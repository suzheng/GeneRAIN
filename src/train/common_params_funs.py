from utils.params import params
params = params()
import torch
import torch.distributed as dist
from utils.config_loader import Config
from utils.json_utils import JsonUtils
import os
import numpy as np
from utils.ParamFinder import ParamFinder
from utils.checkpoint_utils import find_latest_checkpoint


config = Config()
if os.environ.get("RUNNING_MODE") == "debug":
    print("Run in debugging mode!")
    config = Config(config.project_path + "/src/test/config.json")
else:
    print("Run in training mode!")
ju = JsonUtils()

param_json_file = os.environ.get("PARAM_JSON_FILE")
print(f"param_json_file is {param_json_file}")
param_finder = ParamFinder(param_json_file)

# Usage:
h5_file_path = config.get("ARCHS_gene_expression_h5_path")
gene_to_idx_path = config.get("gene2vec_gene_to_idx_json_path")

TENSORBOARD_LOG_DIR_PATH = config.get("tensorboard_log_dir_path")

BASE_SEED = 8

params.NUMBER_OF_SPECIAL_TOKEN_IN_DATASET = 0
NUMBER_OF_SPECIAL_TOKEN = config.get("number_of_special_embeddings")

if False:
    params.GENE_EMB_NAME = param_finder.find("params.GENE_EMB_NAME", "gene2vec") # gene2vec coding_pseudo coding_lncrna coding_smallrna coding_hs_mouse coding 
    params.TOTAL_NUMBER_OF_DATASETS = param_finder.find("params.TOTAL_NUMBER_OF_DATASETS", 5)
    params.DATASET_TO_GET_FOR_MIXED_DATASET = param_finder.find("params.DATASET_TO_GET_FOR_MIXED_DATASET", None) # ["both", "human", "nonhuman", None]
    params.HIDDEN_SIZE = param_finder.find("params.HIDDEN_SIZE", 200)
    model_dim = params.HIDDEN_SIZE
    params.PERFORMER_NET_LAST_LAYER_REQUIRES_GRAD = param_finder.find("params.PERFORMER_NET_LAST_LAYER_REQUIRES_GRAD", True)
    params.FINETUNE_TO_RECONSTRUCT_EXPR_OF_ALL_GENES = param_finder.find("params.FINETUNE_TO_RECONSTRUCT_EXPR_OF_ALL_GENES", False)
    params.USE_PRETRAIN_MODEL_FOR_FINETUNE = param_finder.find("params.USE_PRETRAIN_MODEL_FOR_FINETUNE", True)
    params.PRETRAIN_EXPERIMENT_FOR_FINETUNE = param_finder.find("params.PRETRAIN_EXPERIMENT_FOR_FINETUNE", "exp9")

    #be one of 'Performer', "Traditional_Transformer", "Bert"
    params.TRANSFORMER_MODEL_NAME = param_finder.find("params.TRANSFORMER_MODEL_NAME", "Bert")
    params.LAYER_NORM_EPS = param_finder.find("params.LAYER_NORM_EPS", 1e-12)
    params.OUTPUT_ATTENTIONS = param_finder.find("params.OUTPUT_ATTENTIONS", False)
    params.OUTPUT_HIDDEN_STATES = param_finder.find("params.OUTPUT_HIDDEN_STATES", False)
    params.ONLY_USE_PERTURBED_GENE_TO_PREDICT = param_finder.find("params.ONLY_USE_PERTURBED_GENE_TO_PREDICT", False)

    params.LEARN_ON_ZERO_EXPR_GENES = param_finder.find("params.LEARN_ON_ZERO_EXPR_GENES", False)
    params.OUTPUT_PARAMETER_HIST_TO_TENSOBOARD_BY_BATCH = param_finder.find("params.OUTPUT_PARAMETER_HIST_TO_TENSOBOARD_BY_BATCH", False)
    params.TRANSFORMER_NORM_FIRST = param_finder.find("params.TRANSFORMER_NORM_FIRST", True)
    params.TRANSFORMER_HIDDEN_ACT_FUNC = param_finder.find("params.TRANSFORMER_HIDDEN_ACT_FUNC", "gelu")
    params.MIN_MEAN_VAL_FOR_ZSCORE = param_finder.find("params.MIN_MEAN_VAL_FOR_ZSCORE", 0.1)
    params.SAMPLE_NUMBER_FOR_EACH_PERTURBATION = param_finder.find("params.SAMPLE_NUMBER_FOR_EACH_PERTURBATION", 10)

    params.PERTURBED_GENE_ALWAYS_IN_INPUT_EXPR_IN_PERTURB_DATASET = param_finder.find("params.PERTURBED_GENE_ALWAYS_IN_INPUT_EXPR_IN_PERTURB_DATASET", False)



    params.PRETRAIN_LOSS_ONLY_ON_MASKED_GENES = param_finder.find("params.PRETRAIN_LOSS_ONLY_ON_MASKED_GENES", True)
    params.USE_AND_KEEP_ZERO_EXPR_GENES = param_finder.find("params.USE_AND_KEEP_ZERO_EXPR_GENES", True)
    params.NUM_OF_GENES_SELECTED = param_finder.find("params.NUM_OF_GENES_SELECTED", -1) # -1 for selecting all genes
    params.ONLY_USE_POSITIVE_ZSCORES_IN_TRAINING = param_finder.find("params.ONLY_USE_POSITIVE_ZSCORES_IN_TRAINING", False)

    params.SHUFFLE_GENE_INDICES_IN_EVALUATION = param_finder.find("params.SHUFFLE_GENE_INDICES_IN_EVALUATION", False)
    params.SHUFFLE_EXPR_INDICES_IN_EVALUATION = param_finder.find("params.SHUFFLE_EXPR_INDICES_IN_EVALUATION", False)

    params.METHOD_TO_COMBINE_INPUT_AND_ENCODING = param_finder.find("params.METHOD_TO_COMBINE_INPUT_AND_ENCODING", None)

    params.NUM_BINS = param_finder.find("params.NUM_BINS", 100)

    #fraction of genes whose expression will be masked in each epoch, like at epoch 1 mask 10% of genes, epoch 2 mask 20%
    params.MASK_FRACTIONS = param_finder.find("params.MASK_FRACTIONS", [0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    params.PERCENT_OF_MASKED_GENES_ASSIGNED_AS_TOKEN_ZERO = param_finder.find("params.PERCENT_OF_MASKED_GENES_ASSIGNED_AS_TOKEN_ZERO", 0.8)
    params.PERCENT_OF_MASKED_GENES_ASSIGNED_AS_RANDOM_TOKENS = param_finder.find("params.PERCENT_OF_MASKED_GENES_ASSIGNED_AS_RANDOM_TOKENS", 0.1)

    params.BATCH_SIZE = param_finder.find("params.BATCH_SIZE", 8)

    #number of layers of the model
    params.MODEL_DEPTH = param_finder.find("params.MODEL_DEPTH", 6)

    #number of tranformer attention heads
    params.NUM_HEADS = param_finder.find("params.NUM_HEADS", 8)

    #dimension of the attention heads
    params.DIM_HEAD = param_finder.find("params.DIM_HEAD", 32)

    #whether to use projection to get proximation of attention
    params.NO_RPOJECTION = param_finder.find("params.NO_RPOJECTION", False)

    #whether to use reversible layers
    params.MODEL_REVERSIBLE = param_finder.find("params.MODEL_REVERSIBLE", True)

    # Number of iterations before redrawing features in the FAVOR+ attention mechanism
    params.FEATURE_REDRAW_INTERVAL = param_finder.find("params.FEATURE_REDRAW_INTERVAL", 1000)

    # Dropout probability applied to the output of the embedding layer
    params.EMB_DROPOUT = param_finder.find("params.EMB_DROPOUT", 0.)

    # Dropout probability applied to the output of the feed-forward layers
    params.FF_DROPOUT = param_finder.find("params.FF_DROPOUT", 0.1)

    # Dropout probability applied to the output of the self-attention mechanism
    params.ATTN_DROPOUT = param_finder.find("params.ATTN_DROPOUT", 0.1)

    params.OUTPUTLAYER2FCS_DROPOUT_RATE = param_finder.find("params.OUTPUTLAYER2FCS_DROPOUT_RATE", 0.1)

    # If True, use generalized attention; otherwise, use standard self-attention
    params.GENERALIZED_ATTENTION = param_finder.find("params.GENERALIZED_ATTENTION", False)

    # The type of expression embedding used ("positional" or other options)
    params.EXPRESSION_EMB_TYPE = param_finder.find("params.EXPRESSION_EMB_TYPE", "positional")

    # The type of output layer (in this case, two fully connected layers)
    params.TO_OUT_LAYER_TYPE = param_finder.find("params.TO_OUT_LAYER_TYPE", "2FCs")

    # The hidden size of the first fully connected layer in the output layer
    params.OUTPUT_LAYER_HIDDEN_SIZE1 = param_finder.find("params.OUTPUT_LAYER_HIDDEN_SIZE1", 40)

    # The hidden size of the second fully connected layer in the output layer
    params.OUTPUT_LAYER_HIDDEN_SIZE2 = param_finder.find("params.OUTPUT_LAYER_HIDDEN_SIZE2", 20)

    params.PRETRAINED_TOKEN_EMB_FOR_INIT = param_finder.find("params.PRETRAINED_TOKEN_EMB_FOR_INIT", False)
    # If True, allow the gradients to update the gene ID embedding during training
    params.GENE_ID_EMB_REQUIRES_GRAD = param_finder.find("params.GENE_ID_EMB_REQUIRES_GRAD", True)

    # If True, allow the gradients to update the expression embedding during training
    params.EXPR_EMB_REQUIRES_GRAD = param_finder.find("params.EXPR_EMB_REQUIRES_GRAD", True)

    # Base learning rate for the learning rate scheduler
    params.BASE_LR = param_finder.find("params.BASE_LR", 0.00001)

    # Maximum learning rate for the learning rate scheduler
    params.MAX_LR = param_finder.find("params.MAX_LR", 0.0001)

    params.EPOCH_TO_HAVE_MANUAL_LR = param_finder.find("params.EPOCH_TO_HAVE_MANUAL_LR", 30)

    params.ONE_CYCLE_LR_PCT_START = param_finder.find("params.ONE_CYCLE_LR_PCT_START", 0.2)
    params.ONE_CYCLE_LR_DIV_FACTOR = param_finder.find("params.ONE_CYCLE_LR_DIV_FACTOR", 5)
    params.ONE_CYCLE_LR_TOTAL_STEPS = param_finder.find("params.ONE_CYCLE_LR_TOTAL_STEPS", 40)
    params.ONE_CYCLE_LR_EPOCHS = param_finder.find("params.ONE_CYCLE_LR_EPOCHS", 40)

    # Number of iterations for the learning rate to go from the base to the max learning rate
    params.STEP_SIZE_UP = param_finder.find("params.STEP_SIZE_UP", 4)

    # Method for discretizing expression values
    params.EXPR_DISCRETIZATION_METHOD = param_finder.find("params.EXPR_DISCRETIZATION_METHOD", "Direct_quantile")

    # Fraction of the dataset to be used for training (remaining for validation)
    params.TRAINING_SET_FRACTION = param_finder.find("params.TRAINING_SET_FRACTION", 0.9)
    params.GRADIENT_ACCUMULATION_STEPS = param_finder.find("params.GRADIENT_ACCUMULATION_STEPS", 5)
    params.OPTIMIZER = param_finder.find("params.OPTIMIZER", "AdamW")
    params.ADAMW_WEIGHT_DECAY = param_finder.find("params.ADAMW_WEIGHT_DECAY", 0.01)
    params.LOSS_FN = param_finder.find("params.LOSS_FN", "MSE")
    params.SCHEDULER = param_finder.find("params.SCHEDULER", "OneCycleLR")
    params.SAVE_CHECK_POINT_BY_BATCHES = param_finder.find("params.SAVE_CHECK_POINT_BY_BATCHES", False)

    

PRETRAIN_MODEL_CHECKPOINT_PATH = config.get("checkpoint_dir_path") + "/" + str(param_finder.find("params.SPECIFIED_PRETRAIN_MODEL_CHECKPOINT_PATH", None))
if not os.path.isfile(PRETRAIN_MODEL_CHECKPOINT_PATH):
    PRETRAIN_MODEL_CHECKPOINT_PATH = config.get("checkpoint_dir_path") + f"/pretrain/{params.PRETRAIN_EXPERIMENT_FOR_FINETUNE}/model.rank0."
    print(PRETRAIN_MODEL_CHECKPOINT_PATH)
    PRETRAIN_MODEL_CHECKPOINT_PATH = find_latest_checkpoint(PRETRAIN_MODEL_CHECKPOINT_PATH)
    print(PRETRAIN_MODEL_CHECKPOINT_PATH)
else:
    print(f"Will use specified PRETRAIN_MODEL_CHECKPOINT_PATH {PRETRAIN_MODEL_CHECKPOINT_PATH}")
EMSEMBL2GENE_PATH = config.get("Ensembl_ID_gene_symbol_mapping_file_path")




def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_special_encoding(original_encoding, special_token_emb_module, num_special_tokens):
    batch_size = original_encoding.size(0)
    special_token_indices = torch.arange(num_special_tokens).expand(batch_size, -1).to(original_encoding.device)
    # Step 2: Get embeddings for special tokens
    encoding_of_special_tokens = special_token_emb_module(special_token_indices)
    return encoding_of_special_tokens

def cleanup():
    dist.destroy_process_group()

def normalize_expression(data, num_bins=params.NUM_BINS):
    data = (data - num_bins/2.0)/(num_bins/2.0)
    return data

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    seed = worker_info.seed % (2**32 - 1)  # Ensure the seed is within the acceptable range
    np.random.seed(seed)

def custom_histogram(tensor, bins=40, min=None, max=None):
    if min is None:
        min = np.min(tensor)
    if max is None:
        max = np.max(tensor)
    hist, bin_edges = np.histogram(tensor, bins=bins, range=(min, max))
    return hist, bin_edges

def add_histogram_to_tensorboard(writer, tag, data, epoch):
    flattened_data = data.detach().cpu().numpy().flatten()
    hist, bin_edges = np.histogram(flattened_data, bins=40)
    sum_data = np.sum(flattened_data)
    sum_squares_data = np.sum(flattened_data ** 2)

    writer.add_histogram_raw(
        tag=tag,
        min=float(np.min(flattened_data)),
        max=float(np.max(flattened_data)),
        num=len(flattened_data),
        sum=sum_data,
        sum_squares=sum_squares_data,
        bucket_limits=bin_edges[:-1],
        bucket_counts=hist,
        global_step=epoch
    )

def get_current_learning_rate(optimizer):
    # Assuming a single param group, you can access its learning rate
    return optimizer.param_groups[0]['lr']

def shuffle_sequences_old(tensor):
    """
    Shuffle the sequences for each sample in the batch.

    Parameters:
    - tensor (torch.Tensor): Input tensor of shape [batch_size, seq_len]

    Returns:
    - torch.Tensor: Tensor with shuffled sequences for each batch sample.
    """
    batch_size, seq_len = tensor.shape
    for i in range(batch_size):
        tensor[i] = tensor[i, torch.randperm(seq_len)]
    return tensor

def shuffle_sequences(tensor):
    """
    Shuffle the sequences for each sample in the batch.

    Parameters:
    - tensor (torch.Tensor): Input tensor of shape [batch_size, seq_len]

    Returns:
    - torch.Tensor: Tensor with shuffled sequences for each batch sample.
    """
    batch_size, seq_len = tensor.shape
    shuffled_tensor = tensor.clone()  # Create a copy to keep the original tensor unchanged
    for i in range(batch_size):
        shuffled_tensor[i] = shuffled_tensor[i, torch.randperm(seq_len)]
    return shuffled_tensor

def get_pred_using_model_and_input(model, gene_indices, input_expression, zero_expression_genes=None, transformer_model_name=params.TRANSFORMER_MODEL_NAME, output_attentions=params.OUTPUT_ATTENTIONS, output_hidden_states=params.OUTPUT_HIDDEN_STATES, 
                                   shuffle_gene_indices=False, shuffle_expr_indices=False, only_use_perturbed_gene_to_predict=params.ONLY_USE_PERTURBED_GENE_TO_PREDICT, labels=None, **kwargs):
    """
    Get prediction.

    Args:
    - model: The pre-trained transformer model.
    - transformer_model_name (str): The name of the transformer model.
    - gene_indices: Indices specifying the genes.
    - input_expression: Input expression values.
    - zero_expression_genes: Mask that specifies which genes have zero expression.
    - output_attentions (bool, optional): Whether to output attentions, applicable for 'Bert'. Defaults to False.
    - output_hidden_states (bool, optional): Whether to output hidden states, applicable for 'Bert'. Defaults to False.

    Returns:
    - Tensor containing the predicted expression values.

    Raises:
    - ValueError: If the transformer_model_name does not match any known model.
    """
    if only_use_perturbed_gene_to_predict:
        pred_expression = model.module.to_out(kwargs['down_weighted_gene_emb_sum'])
        return pred_expression
    #print(kwargs)
    if shuffle_gene_indices:
        #print(f"original gene indices {kwargs['down_weighted_gene_emb_sum']}")
        kwargs['down_weighted_gene_emb_sum'] = shuffle_sequences(kwargs['down_weighted_gene_emb_sum'])
        gene_indices = shuffle_sequences(gene_indices)
        #print(f"after shuffling {kwargs['down_weighted_gene_emb_sum']}")
    if shuffle_expr_indices:
        #print(f"original gene expr {input_expression}")
        input_expression = shuffle_sequences(input_expression)
        #print(f"after shuffling {input_expression}")
    if zero_expression_genes == None:
        zero_expression_genes = torch.zeros_like(gene_indices, dtype=torch.bool)
    ## In Performer, to mask out a gene, set them to False
    if transformer_model_name == "GPT":
        #1 for tokens that are not masked, 0 for tokens that are masked
        pred_expression = model(gene_indices, 
                                attention_mask=(~zero_expression_genes).int(),
                                output_attentions=output_attentions,
                                output_hidden_states=output_hidden_states,
                                **kwargs
                                )
    elif transformer_model_name == "Bert_pred_tokens":
        #print("using Bert_pred_tokens")
        #print(f"{kwargs}")
        pred_expression = model(gene_indices, 
                                attention_mask=(~zero_expression_genes).int(),
                                output_attentions=output_attentions,
                                output_hidden_states=output_hidden_states,
                                labels=labels,
                                **kwargs
                                )
    elif transformer_model_name == "BertExprInnerEmb":
        #print("using Bert_pred_tokens")
        #print(f"{kwargs}")
        # swap the token and position embeddings, as the Bert from the huggingface library predicts the tokens.
        pred_expression = model(input_expression, 
                                attention_mask=(~zero_expression_genes).int(),
                                output_attentions=output_attentions,
                                output_hidden_states=output_hidden_states,
                                position_ids=gene_indices,
                                labels=labels,
                                **kwargs
                                )
    
    else:
        raise ValueError(f"Unknown transformer model name: {transformer_model_name}")
    return pred_expression

from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions, MaskedLMOutput
import torch

def extract_hidden_states(output, layer=-1):
    #print(f"len(output.hidden_states): {len(output.hidden_states)}")
    if (not isinstance(output, CausalLMOutputWithCrossAttentions)) and (not isinstance(output, BaseModelOutputWithPoolingAndCrossAttentions)) and (not isinstance(output, MaskedLMOutput)):
        return output
    
    if output.hidden_states is None:
        raise ValueError("Hidden states are not available. Make sure output_hidden_states=True when calling the model")
    
    if layer < 0:
        layer = len(output.hidden_states) + layer
    
    if layer < 0 or layer >= len(output.hidden_states):
        raise IndexError("Layer index out of range")
    
    return output.hidden_states[layer]

def get_layers_in_model(model, transformer_model_name=params.TRANSFORMER_MODEL_NAME):
    if hasattr(model, 'gene_expr_transformer'):
        gene_expr_transformer = model.gene_expr_transformer
    else:
        gene_expr_transformer = model


    layers = None
    if transformer_model_name == "Traditional_Transformer":
         #print(kwargs['src_key_padding_mask'].dtype)
        layers = gene_expr_transformer.transformer.layers
    elif transformer_model_name == "Performer":
        layers = gene_expr_transformer.performer.net.blocks
    elif transformer_model_name in ["Bert", "Bert_pred_tokens", "BertExprInnerEmb"]:
        layers = gene_expr_transformer.bert_model.encoder.layer
    elif transformer_model_name == "GPT":
        layers = gene_expr_transformer.gpt.transformer.h
    else:
        raise ValueError(f"Unknown transformer model name: {transformer_model_name}")
    return layers

def get_gene_symbols_filt_on_z_dup(use_and_keep_zero_expr_genes = params.USE_AND_KEEP_ZERO_EXPR_GENES, min_mean_val_for_zscore = params.MIN_MEAN_VAL_FOR_ZSCORE):
    import pandas as pd
    from utils.utils import get_config
    config = get_config()
    ARCHS_file_basename_prefix = config.get("ARCHS_file_basename_prefix")
    if use_and_keep_zero_expr_genes:
        output_file_prefix = config.proj_path + f"/data/external/ARCHS/normalize_each_gene/{ARCHS_file_basename_prefix}_with_zero_expr_genes"
    else:
        output_file_prefix = config.proj_path + f"/data/external/ARCHS/normalize_each_gene/{ARCHS_file_basename_prefix}_without_zero_expr_genes"
    
    gene_stat_uniq_bool_file = output_file_prefix + ".gene_stat_filt_on_z_dup.tsv"
    df = pd.read_csv(gene_stat_uniq_bool_file, sep='\t')

    # Create a numpy boolean vector from the 'max_mean_nondup' column
    max_mean_nondup_vector = df['max_mean_nondup'].values.astype(bool)

    # Create a numpy boolean vector that indicates whether the 'gene_mean' values are greater than min_mean_val_for_zscore
    gene_mean_vector = (df['gene_mean'] > min_mean_val_for_zscore).values
        # Combine both conditions
    combined_conditions = max_mean_nondup_vector & gene_mean_vector

    # Get the list of gene symbols that meet both conditions
    gene_symbols_filt_on_z_dup = df['gene_symbol'][combined_conditions].tolist()
    return gene_symbols_filt_on_z_dup

def get_gene2idx(gene_emb_name=params.GENE_EMB_NAME, sort_by_index=False):
    from utils.json_utils import JsonUtils
    from utils.utils import get_config
    config = get_config()
    
    ju = JsonUtils()
    gene_to_idx_path = config.get("gene2vec_gene_to_idx_json_path")
    gene_to_idx_path = gene_to_idx_path.replace("gene2vec", gene_emb_name)
    print(f"Reading gene_to_idx_path {gene_to_idx_path}!")
    ori_gene_to_idx = ju.load_data_from_file(gene_to_idx_path)
    if gene_emb_name.startswith("hm_"):
        gene_to_idx = ori_gene_to_idx
    else:
        gene_symbols_filt_on_z_dup = get_gene_symbols_filt_on_z_dup()
        gene_to_idx = {k: v for k, v in ori_gene_to_idx.items() if k in gene_symbols_filt_on_z_dup}
    # Sort the dictionary by values (gene indices) if the option is enabled
    if sort_by_index:
        gene_to_idx = dict(sorted(gene_to_idx.items(), key=lambda item: item[1]))
        print("Sorted the dicts by index!")
    idx_to_gene = {v: k for k, v in gene_to_idx.items()}
    return gene_to_idx, idx_to_gene

def get_gene2idx_of_whole_gene_emb(gene_emb_name=params.GENE_EMB_NAME):
    from utils.json_utils import JsonUtils
    from utils.utils import get_config
    config = get_config()
    
    ju = JsonUtils()
    gene_to_idx_path = config.get("gene2vec_gene_to_idx_json_path")
    gene_to_idx_path = gene_to_idx_path.replace("gene2vec", gene_emb_name)
    print(f"Reading gene_to_idx_path {gene_to_idx_path}!")
    gene_to_idx = ju.load_data_from_file(gene_to_idx_path)
    return gene_to_idx



def get_gene2idx_no_special_token():
    return None
    from utils.json_utils import JsonUtils
    ju = JsonUtils()
    from utils.utils import get_config
    config = get_config()
    gene_to_idx_path = config.get("gene2vec_gene_to_idx_json_no_special_emb_path")
    ori_gene_to_idx = ju.load_data_from_file(gene_to_idx_path)
    gene_symbols_filt_on_z_dup = get_gene_symbols_filt_on_z_dup()
    gene_to_idx = {k: v for k, v in ori_gene_to_idx.items() if k in gene_symbols_filt_on_z_dup}
    idx_to_gene = {v: k for k, v in gene_to_idx.items()}
    return gene_to_idx, idx_to_gene

gene_to_idx, _ = get_gene2idx()
gene2idx_of_whole_gene_emb = get_gene2idx_of_whole_gene_emb()
