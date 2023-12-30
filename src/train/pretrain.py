from utils.params import params
params = params()
from train.common_params_funs import TENSORBOARD_LOG_DIR_PATH, PRETRAIN_MODEL_CHECKPOINT_PATH, BASE_SEED, config, set_seed, cleanup, worker_init_fn, get_current_learning_rate, get_layers_in_model
from train.common import train, evaluate, output_parameter_hist_to_tensorboard, initiate_model

from data.ARCHSDataset import ARCHSDataset

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from utils.config_loader import Config
import torch.nn as nn
from torch.utils.data import random_split
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

from utils.json_utils import JsonUtils
import os
import numpy as np
from utils.checkpoint_utils import find_latest_checkpoint


total_number_of_datasets = params.TOTAL_NUMBER_OF_DATASETS

if os.environ.get("RUNNING_MODE") == "debug":
    total_number_of_datasets = 2

from utils.config_loader import Config
config = Config()
if os.environ.get("RUNNING_MODE") == "debug":
    print("Run in debugging mode!")
    config = Config(config.project_path + "/src/test/config.json")
else:
    print("Run in training mode!")

h5_file_path = config.get("ARCHS_gene_expression_h5_path")

from utils.ParamFinder import ParamFinder
param_json_file = os.environ.get("PARAM_JSON_FILE")
print(f"param_json_file is {param_json_file}")
param_finder = ParamFinder(param_json_file)


#PRETRAIN_MODEL_CHECKPOINT_PATH = config.get("checkpoint_dir_path") + f"/pretrain/{params.PRETRAIN_EXPERIMENT_FOR_FINETUNE}/model.rank0."

if PRETRAIN_MODEL_CHECKPOINT_PATH != None and (not os.path.isfile(PRETRAIN_MODEL_CHECKPOINT_PATH)):
    PRETRAIN_MODEL_CHECKPOINT_PATH = find_latest_checkpoint(PRETRAIN_MODEL_CHECKPOINT_PATH)

    
def train_ddp(mask_fractions, epochs_for_training, exp_label, mode="pretrain"):
    
    # Initialize the process group with the chosen backend
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"job at rank{rank} started\n")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    if rank == 0:
        param_finder.save_updated_parameters(f'{TENSORBOARD_LOG_DIR_PATH}/{mode}/{exp_label}/{mode}.{exp_label}.params.json')
    model = initiate_model()

    # Create the DDP model
    model.to(device)
    #ddp_model = DDP(model, device_ids=[rank])
    find_unused_parameters = (params.TRANSFORMER_MODEL_NAME == "Bert") or (params.TRANSFORMER_MODEL_NAME == "Performer" and params.FINETUNE_TO_RECONSTRUCT_EXPR_OF_ALL_GENES)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=find_unused_parameters)
    params_to_update = []
    if params.FINETUNE_TO_RECONSTRUCT_EXPR_OF_ALL_GENES:
        print("params.FINETUNE_TO_RECONSTRUCT_EXPR_OF_ALL_GENES")
        params_to_update.extend(ddp_model.module.to_out.parameters())
        for param in ddp_model.module.gene_expr_transformer.parameters():
            param.requires_grad = False
        if params.PERFORMER_NET_LAST_LAYER_REQUIRES_GRAD:
            print("params.PERFORMER_NET_LAST_LAYER_REQUIRES_GRAD == True")
            last_layer_params = (get_layers_in_model(ddp_model.module)[-1]).parameters()
            for param in last_layer_params:
                param.requires_grad = True
            params_to_update.extend(last_layer_params)
    else:
        params_to_update.extend(ddp_model.parameters())
    if params.TRANSFORMER_MODEL_NAME != "Performer" or params.FINETUNE_TO_RECONSTRUCT_EXPR_OF_ALL_GENES:
        if params.OPTIMIZER == "AdamW":
            print("use AdamW optimizer!")
            optimizer = optim.AdamW(params_to_update, lr=params.MAX_LR/5.0, weight_decay=params.ADAMW_WEIGHT_DECAY)
        if params.OPTIMIZER == "Adam":
            print("use Adam optimizer!")
            optimizer = optim.Adam(params_to_update, lr=params.MAX_LR/5.0)
    # as performer has reversible, so it can't use params_to_update
    else:
        if params.OPTIMIZER == "AdamW":
            print("use AdamW optimizer!")
            optimizer = optim.AdamW(ddp_model.parameters(), lr=params.MAX_LR/5.0, weight_decay=params.ADAMW_WEIGHT_DECAY)
        if params.OPTIMIZER == "Adam":
            print("use Adam optimizer!")
            optimizer = optim.Adam(ddp_model.parameters(), lr=params.MAX_LR/5.0)        
    if params.LOSS_FN == "MSE":
        loss_fn = nn.MSELoss()
    if params.SCHEDULER == "CyclicLR":
        scheduler = CyclicLR(optimizer, base_lr=params.BASE_LR, max_lr=params.MAX_LR, step_size_up=params.STEP_SIZE_UP, gamma=0.8, cycle_momentum=False, mode='triangular2',  last_epoch=-1)
    elif params.SCHEDULER == "OneCycleLR":
        print("Use params.SCHEDULER of OneCycleLR")
        scheduler = OneCycleLR(optimizer, max_lr=params.MAX_LR, pct_start=params.ONE_CYCLE_LR_PCT_START, div_factor=params.ONE_CYCLE_LR_DIV_FACTOR, total_steps=params.ONE_CYCLE_LR_TOTAL_STEPS, epochs=params.ONE_CYCLE_LR_EPOCHS)
    # Training loop
    model_loaded = False
    for epoch in epochs_for_training:
        if epoch == 0 and params.USE_PRETRAIN_MODEL_FOR_FINETUNE and params.FINETUNE_TO_RECONSTRUCT_EXPR_OF_ALL_GENES:
            #only load model, but not optimizer and scheduler
            ddp_model, _, _ = load_checkpoint(ddp_model, None, PRETRAIN_MODEL_CHECKPOINT_PATH, None)
    
        if epochs_for_training[0] != 0 and model_loaded == False:
            ddp_model, optimizer, scheduler = load_checkpoint(ddp_model, optimizer, config.get("checkpoint_dir_path") + f"/{mode}/{exp_label}/model.rank0.epoch{epoch}.pth", scheduler)
            if epoch < params.EPOCH_TO_HAVE_MANUAL_LR:
                scheduler.step()
            model_loaded = True
        if epoch < len(mask_fractions):
            mask_fraction = mask_fractions[epoch]
        else:
            mask_fraction = mask_fractions[-1]
        #to make sure different epochs will have different masking
        
        previous_batches_size = 0
        total_train_loss = 0.0
        total_train_samples = 0
        total_val_loss = 0.0
        total_val_samples = 0
        for dataset_idx in range(0, total_number_of_datasets):
            set_seed(BASE_SEED + epoch + dataset_idx * 1000)

            dataset = ARCHSDataset(h5_file_path, 
                                   n_bins=params.NUM_BINS,
                                   mask_fraction=mask_fraction, 
                                   expr_discretization_method = params.EXPR_DISCRETIZATION_METHOD, 
                                   load_data_into_mem=True, 
                                   chunk_idx=dataset_idx,
                                   num_of_genes=params.NUM_OF_GENES_SELECTED
                                   )
            #for testing
            if os.environ.get("RUNNING_MODE") == "debug":
                dataset = data.Subset(dataset, list(range(200)))
            # Split dataset into training and validation sets
            train_size = int(params.TRAINING_SET_FRACTION * len(dataset))
            val_size = len(dataset) - train_size
            #to make sure all processed are using the same train validation data split in ALL EPOCHs
            #the means the samples in validation dataset are never used for training
            set_seed(BASE_SEED)
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            set_seed(BASE_SEED + epoch + rank + dataset_idx * 1000)

            train_sampler = DistributedSampler(train_dataset)
            val_sampler = DistributedSampler(val_dataset)
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        
            #to make different epochs have different distributions acorss processes
            #even we set shuffle=False here, the DistributedSampler by default it shuffles the data already
            #this was checked by manual inspect the selected samples, different samples were selected in diff epochs even set_seed(BASE_SEED + epoch + rank) was commented out
            train_loader = DataLoader(train_dataset, batch_size=params.BATCH_SIZE, shuffle=False, sampler=train_sampler, num_workers=12, pin_memory=True, worker_init_fn=worker_init_fn, persistent_workers=True)
            val_loader = DataLoader(val_dataset, batch_size=params.BATCH_SIZE, shuffle=False, sampler=val_sampler, num_workers=12, pin_memory=True, worker_init_fn=worker_init_fn, persistent_workers=True)
            param_by_batches_log_dir = f'{TENSORBOARD_LOG_DIR_PATH}/param_by_batches_{mode}/{exp_label}/epoch{epoch+1}/dataset{dataset_idx}/rank_{rank}'
            checkpoint_by_batches_dir = config.get("checkpoint_dir_path") + f"/by_batches_{mode}/{exp_label}/epoch{epoch+1}/dataset{dataset_idx}/rank_{rank}/"
            eval_data_file_path = config.get("evaluation_data_dir_path") + f"/eval_data_{mode}_{exp_label}_epoch{epoch+1}_dataset{dataset_idx}_rank_{rank}.npz"
            param_by_batches_writer = SummaryWriterAndSaver(param_by_batches_log_dir)
            print(f"\n[Rank {rank}] Training and validation for exp={exp_label}, dataset={dataset_idx}, mask fraction {mask_fraction}, epoch={epoch+1}")
            dist.barrier()
            # use updated optimizer for the next dataset, 
            # but don't use the scheduler, as we would like to keep the learning identical across datasets a same epochs
            train_loss, train_samples, optimizer, _ = train(model=ddp_model, 
                                              data_loader=train_loader, 
                                              loss_fn=loss_fn, 
                                              optimizer=optimizer, 
                                              device=device, 
                                              writer=param_by_batches_writer, 
                                              checkpoint_by_batches_dir=checkpoint_by_batches_dir,
                                              epoch=epoch, 
                                              previous_batches_size=previous_batches_size, 
                                              rank=rank, 
                                              scheduler=scheduler, 
                                              exp_label=exp_label)
            dist.barrier()
            val_loss, val_samples = evaluate(ddp_model, val_loader, loss_fn, device, param_by_batches_writer, epoch, exp_label, eval_data_file_path, rank=rank)
            dist.barrier()
            previous_batches_size += len(train_loader)
            # Reduce the losses and sample counts from all processes
            train_loss_tensor = torch.tensor(train_loss * train_samples, device=device)
            val_loss_tensor = torch.tensor(val_loss * val_samples, device=device)
            train_samples_tensor = torch.tensor(train_samples, device=device)
            val_samples_tensor = torch.tensor(val_samples, device=device)

            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_samples_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_samples_tensor, op=dist.ReduceOp.SUM)

            total_train_loss += train_loss_tensor.item()
            total_train_samples += train_samples_tensor.item()
            total_val_loss += val_loss_tensor.item()
            total_val_samples += val_samples_tensor.item()

            # Compute the reduced average losses
            reduced_train_loss = train_loss_tensor.item() / train_samples_tensor.item()
            reduced_val_loss = val_loss_tensor.item() / val_samples_tensor.item()

            current_lr = get_current_learning_rate(optimizer)
            print(f"Rank {rank}, exp={exp_label}, datachunk={dataset_idx}, Mask fraction {mask_fraction}, Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        #outside of dataset, in a single epoch
        if rank == 0:
            total_train_ave_loss = total_train_loss / total_train_samples
            total_val_ave_loss = total_val_loss / total_val_samples
            print(f"Reduced Average - Exp={exp_label}, All data chunks, Mask fraction {mask_fraction}, Epoch: {epoch + 1}, Train Loss: {total_train_ave_loss:.4f}, Validation Loss: {total_val_ave_loss:.4f}")
            hparam_log_dir = f'{TENSORBOARD_LOG_DIR_PATH}/{mode}/{exp_label}/hparam_epoch{epoch + 1}'
            hparam_writer = SummaryWriterAndSaver(hparam_log_dir)     
        #if hparam_writer is not None:
            all_used_params = param_finder.get_param_dict()
            exp_dict = {"Experiment": exp_label, 
                                "rank": rank, 
                                "learning_rate": current_lr, 
                                "mask_fraction": mask_fraction,
                                **all_used_params
                                }
            hparam_writer.add_hparams(exp_dict, 
                                {"hparam/train_loss": total_train_ave_loss,
                                 "hparam/val_loss": total_val_ave_loss})
            hparam_writer.close()
            reduced_loss_log_dir = f'{TENSORBOARD_LOG_DIR_PATH}/{mode}/{exp_label}/reduced_loss'
            reduced_loss_writer = SummaryWriterAndSaver(reduced_loss_log_dir)
            reduced_loss_writer.add_scalar("Reduced_train_loss", total_train_ave_loss, epoch + 1)
            reduced_loss_writer.add_scalar("Reduced_val_loss", total_val_ave_loss, epoch + 1)
            output_parameter_hist_to_tensorboard(ddp_model, epoch + 1, reduced_loss_writer)
            reduced_loss_writer.close()
            save_checkpoint(ddp_model.module, optimizer, config.get("checkpoint_dir_path") + f"/{mode}/{exp_label}/model.rank{rank}.epoch{epoch + 1}.pth", scheduler)
        if epoch < params.EPOCH_TO_HAVE_MANUAL_LR:
            scheduler.step()
        param_by_batches_writer.close()

    cleanup()


def main():
    parser = argparse.ArgumentParser(description='Pretrain DDP Example')
    parser.add_argument('--epoch_from', type=int, default=None, help='starting epoch number of training, starting from 1')
    parser.add_argument('--epoch_to', type=int, default=None, help='ending epoch number of training')
    parser.add_argument('--exp_label', type=str, default=None, help='experiment label for output')
    args = parser.parse_args()
    epochs_for_training = list(range(args.epoch_from-1, args.epoch_to))
    train_ddp(params.MASK_FRACTIONS, epochs_for_training, args.exp_label)

if __name__ == '__main__':
    main()
