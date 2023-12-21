import numpy as np

def get_top_genes(expression_vector_float32, num_of_genes_to_return, only_use_postive_zscores_in_training, index_will_always_be_included=None):
    if num_of_genes_to_return >= len(expression_vector_float32):
        return expression_vector_float32, np.arange(len(expression_vector_float32))
    
    # Get indices of top highest absolute values
    if only_use_postive_zscores_in_training:
        abs_expr = expression_vector_float32.copy()
    else:
        abs_expr = np.abs(expression_vector_float32.copy())
    
    # If index_will_always_be_included is provided, set their expression values to a very large number
    if index_will_always_be_included is not None:
        # If index_will_always_be_included is a single integer, convert it to a list
        if isinstance(index_will_always_be_included, int):
            index_will_always_be_included = [index_will_always_be_included]
        
        # Set the expression values to a very large number
        for idx in index_will_always_be_included:
            abs_expr[idx] = float('inf')
    
    # Get the top indices
    top_num = -1 * num_of_genes_to_return
    top_abs_indices = np.argpartition(abs_expr, top_num)[top_num:]
    
    # Revert the expression values back to their original ones
    expression_vector_float32 = expression_vector_float32[top_abs_indices]
    return expression_vector_float32, top_abs_indices

import numpy as np

def find_indices_of_elements_in_an_array(array, elements, fill_none_for_not_found=False):
    # If elements is a single number, convert it to a list
    if not isinstance(elements, (list, np.ndarray)):
        elements = [elements]
    
    # Initialize a list to store indices
    indices_list = []
    
    # Loop through each element and find its index in the array
    for element in elements:
        indices = np.where(array == element)
        indices = indices[0]
        
        # Check if the element was found in the array
        if indices.size > 0:
            # If the element exists in the array, append the first index where it is located to the list
            indices_list.append(indices[0])
        elif fill_none_for_not_found:
            # If the element does not exist in the array, append None to the list
            indices_list.append(None)
    indices_list = np.array(indices_list)
    return indices_list