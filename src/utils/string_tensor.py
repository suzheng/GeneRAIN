import torch

def string_to_tensor(my_string):
    # For each character in the string, get the ASCII value and create a tensor out of it.
    my_list = [torch.tensor([ord(c)], dtype=torch.int) for c in my_string]
    # Combine tensors
    my_tensor = torch.cat(my_list, dim=0)
    return my_tensor

def tensor_to_string(my_tensor):
    # Convert tensor to list of ASCII values
    ascii_list = my_tensor.tolist()
    # Convert list of ASCII values to a string
    my_string = ''.join([chr(c) for c in ascii_list])
    return my_string