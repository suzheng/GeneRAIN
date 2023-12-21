import torch
import os
import glob
import re

def save_checkpoint(model, optimizer, checkpoint_file, scheduler=None):
    # Get the directory of the checkpoint file
    dir_path = os.path.dirname(checkpoint_file)
    # Create the directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if scheduler is not None:
        checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint_data, checkpoint_file)
    print(f"Model checkpoint saved to: {checkpoint_file}")

def load_partial_model_from_checkpoint(model, checkpoint_state_dict):

    # Obtain the state dict of the current model
    model_state_dict = model.state_dict()

    # Create a new state dict in which we'll load the weights
    new_state_dict = {}

    # Iterate through the current model's state dict
    for layer_name, layer_weights in model_state_dict.items():
        # If this layer exists in the checkpoint and the weights size matches, load it
        if layer_name in checkpoint_state_dict and checkpoint_state_dict[layer_name].shape == layer_weights.shape:
            new_state_dict[layer_name] = checkpoint_state_dict[layer_name]
        else:
            print(f"{layer_name} couldn't be found or has different shape in checkpoint file, will not load")
            # Otherwise, keep the current model's initialized weights
            new_state_dict[layer_name] = layer_weights

    # Load the new state dict into the model
    model.load_state_dict(new_state_dict)


def load_checkpoint(ddp_model, optimizer, checkpoint_file, scheduler=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    checkpoint_state_dict = checkpoint['model_state_dict']

    # Check if model has been parallelized with DDP
    if isinstance(ddp_model, torch.nn.parallel.DistributedDataParallel):
        try:
            ddp_model.module.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Couldn't load complete model state: {e}")
            print("Attempting to load a partial model state...")
            load_partial_model_from_checkpoint(ddp_model.module, checkpoint_state_dict)
    else:
        try:
            ddp_model.load_state_dict(checkpoint_state_dict)
        except Exception as e:
            print(f"Couldn't load complete model state: {e}")
            print("Attempting to load a partial model state...")
            load_partial_model_from_checkpoint(ddp_model, checkpoint_state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"Model checkpoint loaded from: {checkpoint_file}")
    return ddp_model, optimizer, scheduler


def find_latest_checkpoint(checkpoint_prefix):
    
    checkpoint_files = glob.glob(checkpoint_prefix + '*.pth')
    if not checkpoint_files:
        return None
    latest_file = max(checkpoint_files, key=lambda x: int(re.findall(r'epoch(\d+)\.pth', x)[0]))
    print(f"checkpoint file {latest_file} found.")
    return latest_file
