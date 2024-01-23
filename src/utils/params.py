
import os
import json

default_values = {
    "GENE_EMB_NAME": "gene2vec", # gene2vec coding_pseudo coding_lncrna coding_smallrna coding_hs_mouse coding
    "TOTAL_NUMBER_OF_DATASETS": 5,
    "DATASET_TO_GET_FOR_MIXED_DATASET": None, # ["both", "human", "nonhuman", None]
    "HIDDEN_SIZE": 200,
    "PERFORMER_NET_LAST_LAYER_REQUIRES_GRAD": True,
    "FINETUNE_TO_RECONSTRUCT_EXPR_OF_ALL_GENES": False,
    "USE_PRETRAIN_MODEL_FOR_FINETUNE": True,
    "PRETRAIN_EXPERIMENT_FOR_FINETUNE": "exp9",
    "TRANSFORMER_MODEL_NAME": "Bert",
    "LAYER_NORM_EPS": 1e-12,
    "OUTPUT_ATTENTIONS": False,
    "OUTPUT_HIDDEN_STATES": False,
    "ONLY_USE_PERTURBED_GENE_TO_PREDICT": False,
    "LEARN_ON_ZERO_EXPR_GENES": False,
    "OUTPUT_PARAMETER_HIST_TO_TENSOBOARD_BY_BATCH": False,
    "TRANSFORMER_NORM_FIRST": True,
    "TRANSFORMER_HIDDEN_ACT_FUNC": "gelu",
    "MIN_MEAN_VAL_FOR_ZSCORE": 0.1,
    "SAMPLE_NUMBER_FOR_EACH_PERTURBATION": 10,
    "PERTURBED_GENE_ALWAYS_IN_INPUT_EXPR_IN_PERTURB_DATASET": False,
    "PRETRAIN_LOSS_ONLY_ON_MASKED_GENES": True,
    "USE_AND_KEEP_ZERO_EXPR_GENES": True,
    "NUM_OF_GENES_SELECTED": -1, # -1 for selecting all genes
    "ONLY_USE_POSITIVE_ZSCORES_IN_TRAINING": False,
    "SHUFFLE_GENE_INDICES_IN_EVALUATION": False,
    "SHUFFLE_EXPR_INDICES_IN_EVALUATION": False,
    "METHOD_TO_COMBINE_INPUT_AND_ENCODING": None,
    "NUM_BINS": 100,
    "MASK_FRACTIONS": [0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "PERCENT_OF_MASKED_GENES_ASSIGNED_AS_TOKEN_ZERO": 0.8,
    "PERCENT_OF_MASKED_GENES_ASSIGNED_AS_RANDOM_TOKENS": 0.1,
    "BATCH_SIZE": 8,
    "MODEL_DEPTH": 6,
    "NUM_HEADS": 8,
    "DIM_HEAD": 32,
    "NO_RPOJECTION": False,
    "MODEL_REVERSIBLE": True,
    "FEATURE_REDRAW_INTERVAL": 1000,
    "EMB_DROPOUT": 0.,
    "FF_DROPOUT": 0.1,
    "ATTN_DROPOUT": 0.1,
    "OUTPUTLAYER2FCS_DROPOUT_RATE": 0.1,
    "GENERALIZED_ATTENTION": False,
    "EXPRESSION_EMB_TYPE": "positional",
    "TO_OUT_LAYER_TYPE": "2FCs",
    "OUTPUT_LAYER_HIDDEN_SIZE1": 40,
    "OUTPUT_LAYER_HIDDEN_SIZE2": 20,
    "PRETRAINED_TOKEN_EMB_FOR_INIT": False,
    "GENE_ID_EMB_REQUIRES_GRAD": True,
    "EXPR_EMB_REQUIRES_GRAD": True,
    "BASE_LR": 0.00001,
    "MAX_LR": 0.0001,
    "EPOCH_TO_HAVE_MANUAL_LR": 30,
    "ONE_CYCLE_LR_PCT_START": 0.2,
    "ONE_CYCLE_LR_DIV_FACTOR": 5,
    "ONE_CYCLE_LR_TOTAL_STEPS": 40,
    "ONE_CYCLE_LR_EPOCHS": 40,
    "STEP_SIZE_UP": 4,
    "EXPR_DISCRETIZATION_METHOD": "Direct_quantile",
    "TRAINING_SET_FRACTION": 0.9,
    "GRADIENT_ACCUMULATION_STEPS": 5,
    "OPTIMIZER": "AdamW",
    "ADAMW_WEIGHT_DECAY": 0.01,
    "LOSS_FN": "MSE",
    "SCHEDULER": "OneCycleLR",
    "SAVE_CHECK_POINT_BY_BATCHES": False,
    "FRACTION_OF_SAMPLES_TO_BE_FAKE": 0.5, 
    "FRACTION_OF_GENES_TO_HAVE_RANDOM_EXPR": 0.3,
    "SPECIFIED_PRETRAIN_MODEL_CHECKPOINT_PATH": None,
    "NUMBER_OF_SPECIAL_TOKEN_IN_DATASET": 0
}
class params:
    def __init__(self):
        self.init_obj()

    def init_obj(self): 
        self.json_file = os.environ.get("PARAM_JSON_FILE")
        self.default_values = default_values
        try:
            with open(self.json_file, 'r') as file:
                self.data = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"The JSON file '{self.json_file}' does not exist.")
        except json.JSONDecodeError:
            raise ValueError(f"The JSON file '{self.json_file}' is not valid JSON.")     

    def __getattr__(self, item):
        self.init_obj()
        if self.data is None:
            self.data = {}
        if item in self.default_values:
            ret_val = self.default_values.get(item)
        if item in self.data:
            ret_val = self.data.get(item)
            
        # if item in self.default_values and item in self.data and (self.default_values.get(item) != self.data.get(item)):
        #     print(f"Using value of {ret_val} for {item}, which is different from default value of {self.default_values.get(item)}.")

        # if item in self.default_values and item not in self.data:
        #     print(f"Using default value of {ret_val} for {item}.")  
        return ret_val