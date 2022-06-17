import argparse
import torch

from model.smooth_cross_entropy import smooth_crossentropy,trades_loss
from utils.cifar import Cifar,Cifar100
from utils.log import Log
from utils.initialize import initialize
from utils.step_lr import StepLR
from utils.Esam import ESAM
from utils.Esam_vary import ESAM_vary
from utils.sam import SAM
from utils.others import *
from torch.utils.tensorboard import SummaryWriter
import os 
import wandb
import time

from utils.options import args,setup_model
from utils.MiscTools import count_parameters
from utils.dist_util import get_world_size

import torch.nn.functional as F
import logging
from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)

def train(args,model):
    
    wandb_init(args)
    initialize(args, seed=42)
    device = args.device

    dataset = Cifar(args) if args.dataset =="cifar10" else Cifar100(args)
    log = Log(log_each=10)

    if args.SCE_loss =="True":
        loss_fct = smooth_crossentropy
    else:
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    paras = model.parameters()
    base_optimizer = torch.optim.SGD(model.parameters(),lr=args.learning_rate,momentum=0.9,weight_decay=args.weight_decay)
    if(args.mode == 'esam'):
        #optimizer = ESAM_vary(paras, base_optimizer, rho=args.rho, beta=args.beta,gamma=args.gamma,adaptive=args.isASAM,nograd_cutoff=args.nograd_cutoff)
        optimizer = ESAM(paras, base_optimizer, rho=args.rho, beta=args.beta,gamma=args.gamma,adaptive=args.isASAM,nograd_cutoff=args.nograd_cutoff)
        optimizer0 = optimizer.base_optimizer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer0, T_max=args.epochs)
    elif(args.mode == 'sam'):
        optimizer = SAM(paras, base_optimizer, rho=args.rho)
        optimizer0 = optimizer.base_optimizer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer0, T_max=args.epochs)
    elif(args.mode == 'sgd'):
        optimizer = base_optimizer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    #half float setting 
    if args.fp16:
        opt_list = [optimizer0,optimizer]

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model,device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)

    best_acc = 0.0
    global_step = -1
    sampler = dataset.train.sampler
    
    
    for epoch in range(args.epochs):
        
        epoch_train_correct = 0.
        epoch_train_loss = 0.
        epoch_train_len = len(dataset.train.dataset)
        test_len = len(dataset.test.dataset)
        
        if args.local_rank != -1:
            sampler.set_epoch(epoch)
        model.train()
        log.train(len_dataset=len(dataset.train))

        for batch in dataset.train:
            global_step += 1
            inputs, targets = (b.to(args.device) for b in batch)

            def defined_backward(loss):
                if args.fp16:
                    a=0
                else:
                    loss.backward()

                    
                    
            if(args.mode == 'esam'):
                paras = [inputs,targets,loss_fct,model,defined_backward]
                optimizer.paras = paras
                optimizer.step()
                predictions,loss, instance_sharpness = optimizer.returnthings
                
            elif(args.mode == 'sam'):
                logits = model(inputs)
                loss = loss_fct(logits, targets)
                
                predictions = logits
                return_loss = loss.clone().detach()
                loss = loss.mean()
                defined_backward(loss)
                optimizer.first_step(zero_grad=True)
                
                loss = loss_fct(model(inputs), targets)
                loss = loss.mean()
                defined_backward(loss)
                optimizer.second_step(zero_grad=True)
                predictions, loss = predictions, return_loss
                
            elif(args.mode == 'sgd'):
                logits = model(inputs)
                loss = loss_fct(logits, targets)
                
                predictions = logits
                return_loss = loss.clone().detach()
                loss = loss.mean()
                defined_backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                predictions, loss = predictions, return_loss


 

            with torch.no_grad():
                if len(inputs)!=len(predictions):
                    continue
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.get_last_lr()[0])
                acc = (correct.sum()+0.01) / (len(targets)+0.01) 
                
                epoch_train_correct += correct.sum()
                epoch_train_loss += loss.mean() * len(targets)
                
            if  args.local_rank in [-1, 0]:
                writer.add_scalar("train/loss", scalar_value=loss.mean(), global_step=global_step)
                writer.add_scalar("train/acc", scalar_value=acc, global_step=global_step)

        scheduler.step()
        if  args.local_rank in [-1, 0]:
            model.eval()
            log.eval(len_dataset=len(dataset.test))

            with torch.no_grad():
                tol_cor = 0
                tol_len = 0
                for batch in dataset.test:
                    inputs, targets = (b.to(device) for b in batch)

                    predictions = model(inputs)
                    loss = smooth_crossentropy(predictions, targets)
                    correct = torch.argmax(predictions, 1) == targets
                    log(model, loss.cpu(), correct.cpu())
                    acc = (correct.sum()+0.01) / (len(targets)+0.01) 
                    tol_len += len(targets)
                    tol_cor += correct.sum()
                acc = tol_cor/(tol_len+0.0)
                if acc > best_acc:
                    best_acc = acc 
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(),"../output/"+"%s_checkpoint.bin" %args.name)
                writer.add_scalar("test/acc", scalar_value=tol_cor/(tol_len+0.0), global_step=global_step)
                
            epoch_train_acc = epoch_train_correct / epoch_train_len
            epoch_train_loss = epoch_train_loss / epoch_train_len
            elapsed_time = log._time()
            wandb_log_some(args, ['train_loss', 'train_acc', 'test_acc', 'elapsed_time'], [epoch_train_loss.item(), epoch_train_acc.item(), acc.item(), float(elapsed_time)])
                
    if args.local_rank in [-1,0]:
        
        log.flush()

def main(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    train_name = "train" 
    log_path = args.name + "_" + train_name
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',filename = '../output/logs/'+log_path,
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed

    # Model & Tokenizer Setup
    model = setup_model(args)


    # Training
    train(args, model)


if __name__ == "__main__":
    main(args)


