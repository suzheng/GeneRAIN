from utils.params import params
params = params()
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from utils.config_loader import Config
from utils.json_utils import JsonUtils
import torch.nn as nn
from torch.utils.data import random_split
import os
import numpy as np
import time
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from utils.checkpoint_utils import save_checkpoint, load_checkpoint
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import OneCycleLR
import argparse
from utils.config_loader import Config
from utils.ParamFinder import ParamFinder
from utils.SummaryWriterAndSaver import SummaryWriterAndSaver
from data.adata import Adata
from models.GeneExprTransformerGPT import GeneExprTransformerGPT
from models.GeneExprTransformerBertPredTokens import GeneExprTransformerBertPredTokens
from models.GeneExprTransformerBertExprInnerEmb import GeneExprTransformerBertExprInnerEmb
from utils.utils import print_config_assignments
from train.common_params_funs import config, add_histogram_to_tensorboard, get_current_learning_rate, get_pred_using_model_and_input, get_gene2idx, get_gene2idx_of_whole_gene_emb



#for debugging
def output_to_a_file(masked_idx, output_file = config.project_path + '/results/debugging/others/masked_idx_output.txt'):
    # Convert the tensor 'masked_idx' to a NumPy array
    masked_idx_np = masked_idx.cpu().numpy()
    # Save the NumPy array to the output file
    with open(output_file, 'a') as outfile:
        for row in masked_idx_np:
            outfile.write(' '.join([str(idx) for idx in row]))
            outfile.write('\n')

# Training function
def train(model, data_loader, loss_fn, optimizer, device, writer, checkpoint_by_batches_dir, epoch, previous_batches_size, rank, scheduler, exp_label, mode="pretrain", print_every_n_batches=10, gradient_accumulation_steps=params.GRADIENT_ACCUMULATION_STEPS, times_output_to_tensorboard=6, save_check_point_by_batches=True):
    #how many output to tensorflow in one epoch
    num_of_batch_to_output_to_tensorboard = len(data_loader) // times_output_to_tensorboard
    if num_of_batch_to_output_to_tensorboard == 0:
        num_of_batch_to_output_to_tensorboard = 1
    model.train()
    total_loss = 0
    num_samples = 0
    start_time = time.time()
    optimizer.zero_grad()
    time_to_output_hist_and_checkpoint = False
    lr = get_current_learning_rate(optimizer)
    for batch_idx, batch in enumerate(data_loader):
        #with autocast():
        
        gene_indices = batch['gene_indices'].to(device)
        labels = None
        ori_zero_expression_genes = batch['zero_expression_genes'].to(device).bool()
        if params.LEARN_ON_ZERO_EXPR_GENES:
            zero_expression_genes = torch.zeros_like(ori_zero_expression_genes, dtype=torch.bool).to(device)
        else:
            zero_expression_genes = ori_zero_expression_genes
        
        if mode == "pretrain":
            input_expression = batch['masked_expression'].to(device)
            true_expression = batch['true_expression'].float().to(device)
            # cal loss using all the non-zero expressed genes, not only the expr masked genes
            masked_booleans = batch['masked_booleans'].to(device)
            if params.PRETRAIN_LOSS_ONLY_ON_MASKED_GENES:
                masked_idx = (masked_booleans & (~zero_expression_genes))
            else:
                masked_idx = (~zero_expression_genes)
            

            if params.TRANSFORMER_MODEL_NAME == "Bert_pred_tokens":
                labels = gene_indices.clone()
                gene_indices[masked_idx] = 0
                labels[~masked_idx] = -100
                labels = labels.to(device)
            elif params.TRANSFORMER_MODEL_NAME == "BertExprInnerEmb":
                labels = true_expression.clone()
                labels[~masked_idx] = -100
                labels = labels.to(device)

            down_weighted_gene_emb_sum = None
            up_weighted_gene_emb_sum = None
            true_expression_of_masked_genes = (true_expression[masked_idx] - params.NUM_BINS/2)/(params.NUM_BINS/2)
            #mask = None
        elif mode == "finetune":
            input_expression = batch['input_binned_expr'].to(device)
            true_expression = batch['output_binned_expr'].float().to(device)
            masked_idx = ((true_expression != -1) & (~zero_expression_genes))
            perturbed_gene_index = batch['perturbed_gene_index'].to(device)
            down_weighted_gene_emb_sum = model.module.gene_expr_transformer.token_emb(perturbed_gene_index)
            down_weighted_gene_emb_sum = down_weighted_gene_emb_sum.squeeze(1)
            up_weighted_gene_emb_sum = None
            true_expression_of_masked_genes = ((true_expression[masked_idx] - input_expression[masked_idx]))/(params.NUM_BINS/2)

        

        if rank == 0 and epoch == 0 and batch_idx == 0:
            tmp = 0
            #writer.add_graph(model, (gene_indices, input_expression))

        # Print the loss after every 'print_every_n_batches' batches
        # It is not global anymore, just the batch index within a single epoch
        global_batch_idx = previous_batches_size + batch_idx

        if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(data_loader) - 1:
            pred_expression = get_pred_using_model_and_input(model, gene_indices, input_expression, zero_expression_genes, 
                                                            transformer_model_name=params.TRANSFORMER_MODEL_NAME, 
                                                            output_attentions=params.OUTPUT_ATTENTIONS, 
                                                            output_hidden_states=params.OUTPUT_HIDDEN_STATES,
                                                            down_weighted_gene_emb_sum=down_weighted_gene_emb_sum,
                                                            up_weighted_gene_emb_sum=up_weighted_gene_emb_sum,
                                                            labels=labels
                                                            )
            if params.TRANSFORMER_MODEL_NAME in ["GPT", "Bert_pred_tokens", "BertExprInnerEmb"]:
                loss = pred_expression.loss
            else:
                pred_expression_of_masked_genes = pred_expression[masked_idx] - (input_expression[masked_idx] - params.NUM_BINS/2)/(params.NUM_BINS/2)
                loss = loss_fn(pred_expression_of_masked_genes.view(-1), true_expression_of_masked_genes.view(-1))
            loss.backward()
            #output hist here, avoid output gradients after optimizer.zero_grad()
            #save it before optimizer.step(), as we would like to save the original model, before any optimization
            if time_to_output_hist_and_checkpoint == True:
                #print("output_parameter_hist_to_tensorboard here!")
                output_parameter_hist_to_tensorboard(model, global_batch_idx, writer)
            optimizer.step()
            if time_to_output_hist_and_checkpoint == True and save_check_point_by_batches == params.SAVE_CHECK_POINT_BY_BATCHES:
                save_checkpoint(model.module, optimizer, f"{checkpoint_by_batches_dir}/model.rank{rank}.batch{global_batch_idx}.pth", scheduler)
                time_to_output_hist_and_checkpoint = False
            optimizer.zero_grad()
        else:
            with model.no_sync():
                pred_expression = get_pred_using_model_and_input(model, gene_indices, input_expression, zero_expression_genes, 
                                                transformer_model_name=params.TRANSFORMER_MODEL_NAME, 
                                                output_attentions=params.OUTPUT_ATTENTIONS, 
                                                output_hidden_states=params.OUTPUT_HIDDEN_STATES,
                                                down_weighted_gene_emb_sum=down_weighted_gene_emb_sum,
                                                up_weighted_gene_emb_sum=up_weighted_gene_emb_sum,
                                                labels=labels
                                                )
                if params.TRANSFORMER_MODEL_NAME in ["GPT", "Bert_pred_tokens", "BertExprInnerEmb"]:
                    loss = pred_expression.loss
                else:
                    pred_expression_of_masked_genes = pred_expression[masked_idx] - (input_expression[masked_idx] - params.NUM_BINS/2)/(params.NUM_BINS/2)
                    loss = loss_fn(pred_expression_of_masked_genes.view(-1), true_expression_of_masked_genes.view(-1))
                loss.backward()
        
        total_loss += loss.item() * gene_indices.size(0)
        num_samples += gene_indices.size(0)


        if global_batch_idx % print_every_n_batches == 0:
            elapsed_time = time.time() - start_time
            print(f"Experiment {exp_label}, {mode} In Device {device}, Epoch: {epoch+1}, LR: {lr}, Across Datasets Batch: {global_batch_idx + 1}, Within Dataset Batch: {batch_idx}, Loss: {loss.item():.4f}, Elapsed Time: {elapsed_time:.2f} seconds")
        if batch_idx % num_of_batch_to_output_to_tensorboard == 0:
            if writer is not None:
                # Calculate the global batch index
                writer.add_scalar(f"Loss/{mode}_batch", loss.item(), global_batch_idx)
                if mode == "pretrain":
                    time_to_output_hist_and_checkpoint = True
                    

    return total_loss / num_samples, num_samples, optimizer, scheduler

# Evaluation function
def evaluate(model, data_loader, loss_fn, device, writer, epoch, exp_label, eval_data_file_path, mode="pretrain", print_every_n_batches=2, rank=None):
    model.eval()
    total_loss = 0
    num_samples = 0
    start_time = time.time()
    # Create lists to store the data
    gene_indices_list = []
    input_expression_list = []
    true_expression_list = []
    pred_expression_list = []
    masked_idx_list = []
    zero_expression_gene_list = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            gene_indices = batch['gene_indices'].to(device)
            labels = None
            ori_zero_expression_genes = batch['zero_expression_genes'].to(device).bool()
            if params.LEARN_ON_ZERO_EXPR_GENES:
                zero_expression_genes = torch.zeros_like(ori_zero_expression_genes, dtype=torch.bool).to(device)
            else:
                zero_expression_genes = ori_zero_expression_genes
            if mode == "pretrain":
                input_expression = batch['masked_expression'].to(device)
                true_expression = batch['true_expression'].float().to(device)
                # cal loss using all the non-zero expressed genes, not only the expr masked genes
                masked_booleans = batch['masked_booleans'].to(device)
                if params.PRETRAIN_LOSS_ONLY_ON_MASKED_GENES:
                    masked_idx = (masked_booleans & (~zero_expression_genes))
                else:
                    masked_idx = (~zero_expression_genes)

                if params.TRANSFORMER_MODEL_NAME == "Bert_pred_tokens":
                    labels = gene_indices.clone()
                    gene_indices[masked_idx] = 0
                    labels[~masked_idx] = -100
                    labels = labels.to(device)
                elif params.TRANSFORMER_MODEL_NAME == "BertExprInnerEmb":
                    labels = true_expression.clone()
                    labels[~masked_idx] = -100
                    labels = labels.to(device)

                down_weighted_gene_emb_sum = None
                up_weighted_gene_emb_sum = None
                true_expression_of_masked_genes = (true_expression[masked_idx] - params.NUM_BINS/2)/(params.NUM_BINS/2)

                #mask = None
            elif mode == "finetune":
                input_expression = batch['input_binned_expr'].to(device)
                true_expression = batch['output_binned_expr'].float().to(device)
                masked_idx = ((true_expression != -1) & (~zero_expression_genes))
                perturbed_gene_index = batch['perturbed_gene_index'].to(device)
                down_weighted_gene_emb_sum = model.module.gene_expr_transformer.token_emb(perturbed_gene_index)
                down_weighted_gene_emb_sum = down_weighted_gene_emb_sum.squeeze(1)
                up_weighted_gene_emb_sum = None
                true_expression_of_masked_genes = ((true_expression[masked_idx] - input_expression[masked_idx]))/(params.NUM_BINS/2)
                
          
            pred_expression = get_pred_using_model_and_input(model, gene_indices, input_expression, zero_expression_genes, 
                                                transformer_model_name=params.TRANSFORMER_MODEL_NAME, 
                                                output_attentions=params.OUTPUT_ATTENTIONS, 
                                                output_hidden_states=params.OUTPUT_HIDDEN_STATES,
                                                down_weighted_gene_emb_sum=down_weighted_gene_emb_sum,
                                                up_weighted_gene_emb_sum=up_weighted_gene_emb_sum,
                                                shuffle_gene_indices=params.SHUFFLE_GENE_INDICES_IN_EVALUATION,
                                                shuffle_expr_indices=params.SHUFFLE_EXPR_INDICES_IN_EVALUATION,
                                                labels=labels
                                                )
            if params.TRANSFORMER_MODEL_NAME in ["GPT", "Bert_pred_tokens", "BertExprInnerEmb"]:
                loss = pred_expression.loss
            else:
                pred_expression_of_masked_genes = pred_expression[masked_idx] - (input_expression[masked_idx] - params.NUM_BINS/2)/(params.NUM_BINS/2)
                loss = loss_fn(pred_expression_of_masked_genes.view(-1), true_expression_of_masked_genes.view(-1))

            total_loss += loss.item() * gene_indices.size(0)
            num_samples += gene_indices.size(0)
            if (batch_idx + 1) % print_every_n_batches == 0:
                elapsed_time = time.time() - start_time
                print(f"Experiment {exp_label}, Evaluating of {mode} In Device {device}, Epoch: {epoch}, Batch: {batch_idx + 1}, Loss: {loss.item():.4f}, Elapsed Time: {elapsed_time:.2f} seconds")
            
            gene_indices_list.append(gene_indices.cpu().numpy())
            input_expression_list.append(input_expression.cpu().numpy()) 
            true_expression_list.append(true_expression.cpu().numpy())
            if not(params.TRANSFORMER_MODEL_NAME in ["GPT", "Bert_pred_tokens", "BertExprInnerEmb"]): # Append true_expression to the list
                pred_expression_list.append(pred_expression.cpu().numpy())
            else:
                pred_expression_list.append([pred_expression.loss.cpu().numpy()])
            masked_idx_list.append(masked_idx.cpu().numpy())
            zero_expression_gene_list.append(ori_zero_expression_genes.cpu().numpy())
    if rank == 0:
        np.savez(eval_data_file_path,
             gene_indices=np.concatenate(gene_indices_list, axis=0),
             input_expression=np.concatenate(input_expression_list, axis=0),
             true_expression=np.concatenate(true_expression_list, axis=0),
             pred_expression=np.concatenate(pred_expression_list, axis=0),
             masked_idx=np.concatenate(masked_idx_list, axis=0),
             zero_expression_genes=np.concatenate(zero_expression_gene_list, axis=0)
             )
    return total_loss / num_samples, num_samples



def output_parameter_hist_to_tensorboard(model, epoch, writer):
    if params.OUTPUT_PARAMETER_HIST_TO_TENSOBOARD_BY_BATCH:
        #print("output_parameter_hist_to_tensorboard is real!")
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            # Output parameter histogram
            if "weight" in name or "bias" in name:
                add_histogram_to_tensorboard(writer, f"{name}/hist", param, epoch)
            # Output gradient histogram
            if "weight" in name or "bias" in name:
                add_histogram_to_tensorboard(writer, f"{name}/grad_hist", param.grad, epoch)
    else:
        return None

from dataclasses import asdict
from models.GeneExprTransformer import GeneExprTransformerConfig

def initiate_model():
    pretrained_emb_path = None
    if params.PRETRAINED_TOKEN_EMB_FOR_INIT:
        pretrained_emb_path = config.get("gene2vec_embeddings_pyn_path")

    gene2idx_of_whole_gene_emb = get_gene2idx_of_whole_gene_emb()
    vocab_size = max(gene2idx_of_whole_gene_emb.values()) + 1
    # Define config dictionary
    config_dict = {
        "num_tokens": vocab_size,
        "dim": params.HIDDEN_SIZE,
        "depth": params.MODEL_DEPTH,
        "heads": params.NUM_HEADS,
        "reversible": params.MODEL_REVERSIBLE,
        "no_projection": params.NO_RPOJECTION,
        "pretrained_emb_path": pretrained_emb_path,
        "dim_head": params.DIM_HEAD,
        "feature_redraw_interval": params.FEATURE_REDRAW_INTERVAL,
        "emb_dropout": params.EMB_DROPOUT,
        "ff_dropout": params.FF_DROPOUT,
        "attn_dropout": params.ATTN_DROPOUT,
        "generalized_attention": params.GENERALIZED_ATTENTION,
        "expression_emb_type": params.EXPRESSION_EMB_TYPE,
        "gene_id_emb_requires_grad": params.GENE_ID_EMB_REQUIRES_GRAD,
        "expr_emb_requires_grad": params.EXPR_EMB_REQUIRES_GRAD,
        "number_of_bins_for_expression_embedding": params.NUM_BINS,
        "layer_norm_eps": params.LAYER_NORM_EPS,
        "norm_first": params.TRANSFORMER_NORM_FIRST,
        "hidden_act": params.TRANSFORMER_HIDDEN_ACT_FUNC,
        "transformer_model_name": params.TRANSFORMER_MODEL_NAME,
        "vocab_size": vocab_size,  # Assuming vocab_size is the number of unique genes
        "n_positions": params.NUM_OF_GENES_SELECTED,  # Assuming n_positions is the number of genes selected
    }

    # Convert config dictionary to GeneExprTransformerConfig object
    model_config = GeneExprTransformerConfig(**config_dict)
    # print_config_assignments(model_config)
    # Create an instance of the appropriate model
    if params.TRANSFORMER_MODEL_NAME == "GPT":
        model = GeneExprTransformerGPT(model_config)
    elif params.TRANSFORMER_MODEL_NAME == "Bert_pred_tokens":
        model = GeneExprTransformerBertPredTokens(model_config)
    elif params.TRANSFORMER_MODEL_NAME == "BertExprInnerEmb":
        model = GeneExprTransformerBertExprInnerEmb(model_config)
    else:
        model = GeneExprTransformer2Expr(
            config=model_config,
            to_out_layer_type=params.TO_OUT_LAYER_TYPE,
            output_layer_hidden_size1=params.OUTPUT_LAYER_HIDDEN_SIZE1,
            output_layer_hidden_size2=params.OUTPUT_LAYER_HIDDEN_SIZE2,
            OutputLayer2FCs_dropout_rate=params.OUTPUTLAYER2FCS_DROPOUT_RATE,
            method_to_combine_input_and_encoding=params.METHOD_TO_COMBINE_INPUT_AND_ENCODING
        )
    # print(model)
    return model


