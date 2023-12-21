import pandas as pd
import torch
import numpy as np

def dict_to_tsv(data, file_path):
    """
    Save a dictionary to a CSV file. Each key-value pair will be a column.
    
    :param data: The dictionary to be saved.
    :param file_path: The file path where the CSV will be saved.
    """
    processed_data = {}
    
    for key, value in data.items():
        # Convert torch.Tensor to numpy array and then to list
        if isinstance(value, torch.Tensor):
            value = value.numpy().tolist()
        
        # If the value is not a list, convert it to a list
        if not isinstance(value, list):
            value = [value]
        
        processed_data[key] = value
    
    # Find the maximum length of the lists
    max_len = max(len(value) for value in processed_data.values())
    
    # Pad the shorter lists with 'nan'
    for key, value in processed_data.items():
        if len(value) < max_len:
            processed_data[key] = value + ['NaN'] * (max_len - len(value))
    
    # Convert the processed dictionary to a DataFrame and save it to a CSV file
    df = pd.DataFrame.from_dict(processed_data)
    df.to_csv(file_path, index=False, sep="\t")