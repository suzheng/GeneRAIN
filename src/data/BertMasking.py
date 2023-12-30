from utils.params import params
params = params()
import numpy as np
import torch
from train.common_params_funs import *

class BertMasking:
    #def __init__(self, mask_fraction=0.15, mask_prob=0.8, random_token_prob=0.1, num_bins=100, mask_token=0):
    def __init__(self, mask_fraction, mask_prob, random_token_prob, num_bins, mask_token):
        self.mask_fraction = mask_fraction
        self.mask_prob = mask_prob
        self.random_token_prob = random_token_prob
        self.num_bins = num_bins
        self.mask_token = mask_token

    def mask_sequence(self, sequence):
        is_tensor = isinstance(sequence, torch.Tensor)

        # Convert tensor to numpy for processing
        if is_tensor:
            sequence_np = sequence.cpu().numpy().copy()
        else:
            sequence_np = np.array(sequence)

        # Create a mask array of the same shape as sequence
        mask = np.random.rand(sequence_np.shape[0]) < self.mask_fraction

        # Generate random strategy choices for each masked position
        strategy_choices = np.random.rand(mask.sum())
        #print(f"strategy_choices {strategy_choices}")
        # For mask_token
        mask_condition = strategy_choices < self.mask_prob
        mask_indices = np.where(mask)[0][mask_condition]  # Advanced indexing
        sequence_np[mask_indices] = self.mask_token

        # For random token
        random_token_condition = (strategy_choices >= self.mask_prob) & \
                                 (strategy_choices < self.mask_prob + self.random_token_prob)
        random_token_indices = np.where(mask)[0][random_token_condition]  # Advanced indexing
        sequence_np[random_token_indices] = np.random.randint(1, self.num_bins+1, size=random_token_condition.sum())

        # Convert back to tensor if original input was a tensor
        if is_tensor:
            sequence_pt = torch.tensor(sequence_np, dtype=sequence.dtype, device=sequence.device)
            mask = torch.tensor(mask, dtype=torch.bool, device=sequence.device)
            return sequence_pt, mask
        else:
            return sequence_np, mask

def get_bert_masking(mask_fraction=params.MASK_FRACTIONS[0]):
    bert_masking = BertMasking(mask_fraction=mask_fraction,
                                mask_prob=params.PERCENT_OF_MASKED_GENES_ASSIGNED_AS_TOKEN_ZERO, 
                                random_token_prob=params.PERCENT_OF_MASKED_GENES_ASSIGNED_AS_RANDOM_TOKENS,
                                mask_token=0,
                                num_bins=params.NUM_BINS
                                )
    return bert_masking