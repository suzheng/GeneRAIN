import torch
import numpy as np
import h5py

def get_config():
    from utils.config_loader import Config
    config = Config()
    return config

def to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data

def get_model(param_json_file, check_point_path, use_old_to_out=False):
    import os
    if 'PARAM_JSON_FILE' not in os.environ:
        os.environ['PARAM_JSON_FILE'] = param_json_file
    from train.common_params_funs import PRETRAINED_TOKEN_EMB_FOR_INIT, TRANSFORMER_MODEL_NAME
    print(f"TRANSFORMER_MODEL_NAME {TRANSFORMER_MODEL_NAME}")
    from train.common import initiate_model
    from utils.checkpoint_utils import save_checkpoint, load_checkpoint
    config = get_config()
    model = initiate_model()
    if use_old_to_out:
        from models.OutputLayer2FCs import OutputLayer2FCs_old
        from train.common_params_funs import HIDDEN_SIZE, OUTPUT_LAYER_HIDDEN_SIZE1, OUTPUT_LAYER_HIDDEN_SIZE2, OUTPUTLAYER2FCS_DROPOUT_RATE
        model.to_out = OutputLayer2FCs_old(HIDDEN_SIZE, OUTPUT_LAYER_HIDDEN_SIZE1, OUTPUT_LAYER_HIDDEN_SIZE2, OUTPUTLAYER2FCS_DROPOUT_RATE)
    if "/RNA_expr_net/" not in check_point_path:
        check_point_path = config.get("checkpoint_dir_path") + f"/{check_point_path}"
    model, optimizer, scheduler = load_checkpoint(model, None, check_point_path, None)
    return model



def get_gene2idx():
    from train.common_params_funs import get_gene2idx
    return get_gene2idx()

def get_gene2idx_no_special_token():
    from train.common_params_funs import get_gene2idx_no_special_token
    return get_gene2idx_no_special_token()


def get_gene_to_idx():
    return get_gene2idx()

def get_gene2vec():
    config = get_config()
    pretrained_emb_path = config.get("gene2vec_embeddings_pyn_path")
    gene2vec = torch.load(pretrained_emb_path)
    return gene2vec.numpy()

def get_device(rank=0):
    if torch.cuda.is_available():
        # Make sure that the requested rank is within the number of available GPUs
        if rank < torch.cuda.device_count():
            device = torch.device(f"cuda:{rank}")
        else:
            raise ValueError(f"Requested CUDA rank {rank} is not available. Number of CUDA devices: {torch.cuda.device_count()}.")
    else:
        device = torch.device("cpu")
    return device

def print_config_assignments(config):
    for key, value in config.__dict__.items():
        # For non-string values
        if not isinstance(value, str):
            print(f"{key} = {value}")
        # For string values, enclose them in quotes
        else:
            print(f"{key} = '{value}'")

def get_gene_symbols_from_h5(h5_file_path):
    h5_file = h5py.File(h5_file_path, "r")
    gene_symbols_from_h5 = np.array(h5_file['meta']['genes']['symbol'][()])
    gene_symbols_from_h5 = np.array([gene_symbol.decode('utf-8') for gene_symbol in gene_symbols_from_h5])
    return gene_symbols_from_h5