import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import set_grad_enabled
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torchvision.models import vit_h_14, ViT_H_14_Weights

import os
import time
from tqdm import tqdm
import logging
import argparse
import traceback
import yaml
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

from dataset import MyDataset
from utils import loadDataset, saveModel, makeDirectory, setupLogger, \
    savePlot, logConfig, loadConfig, time_synchronized

def parse_args():
    # python train.py --config config.yaml
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--config', help='Configuration file', default='config.yaml', type=str)
    return parser.parse_args()

def train(
    model, optimizer, lr_scheduler, train_dataloader, val_dataloader, min_loss,
    best_epoch, log, scaler, start_epochs, nEpochs, ckptDir, device, mixed_precision):

    n_train_batches = len(train_dataloader)
    n_val_batches = len(val_dataloader)

    for epoch in range(start_epochs + 1, nEpochs + 1):
        epoch_val_acc = epoch_val_loss = epoch_train_loss = epoch_train_acc = 0
        learning_rates = [f"{x['lr']:.2e}" for x in optimizer.param_groups]
        for mode in ['Training', 'Validation']:
            model.to(device)
            if mode == 'Training':
                model.train()
                pbar = tqdm(train_dataloader)
            else:
                model.eval()
                pbar = tqdm(val_dataloader, desc=mode)

            for i, (img, label) in enumerate(pbar):
                img = img.to(device, non_blocking=True)
                img = img.half() if mixed_precision else img.float()
                label = label.to(device, non_blocking=True).flatten()
                with autocast(enabled=mixed_precision), set_grad_enabled(mode=='Training'):
                    # forward propagation
                    predict = model(img)
                    # calculate loss
                    loss = F.cross_entropy(predict, label)
                    if mode == 'Training':
                        epoch_train_loss += loss.item()
                    else:
                        epoch_val_loss += loss.item()
                        
                pred_one_hot = torch.argmax(predict, dim=1)
                if mode == 'Training':
                    # calculate accuracy
                    accuracy = float((pred_one_hot == label).float().mean())
                    epoch_train_acc += accuracy
                    # back propagation
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    model.zero_grad()
                    # Update progress bar
                    pbar_msg = f'Epoch: [{epoch}/{nEpochs}], ' +\
                            f'lr: {learning_rates}, ' +\
                            f'Loss: {epoch_train_loss / (i+1):.2f}, ' +\
                            f'Acc:{epoch_train_acc / (i+1):.2f}'
                    pbar.set_description(pbar_msg)
                else:
                    epoch_val_acc += float((pred_one_hot == label).float().mean())
                    
            # Save best model
            if mode == 'Validation' and (val_loss := epoch_val_loss / n_val_batches) < min_loss:
                min_loss = val_loss
                best_epoch = epoch
                saveModel(model.cpu(), log, os.path.join(ckptDir, 'best.pt'))

        # Save model checkpoint
        if epoch % 10 == 0:
            saveModel(model.cpu(), log, os.path.join(ckptDir, f'epoch_{epoch}.pt'))

        log['train_acc_list'].append(epoch_train_acc / n_train_batches)
        log['train_loss_list'].append(epoch_train_loss / n_train_batches)
        log['val_acc_list'].append(epoch_val_acc / n_val_batches)
        log['val_loss_list'].append(epoch_val_loss / n_val_batches)
        logging.info(f"Epoch:{epoch}, " + \
            f"Train_acc:{log['train_acc_list'][-1]:.4f}, " + \
            f"Val_acc:{log['val_acc_list'][-1]:.4f}, " + \
            f"Train_loss:{log['train_loss_list'][-1]:.3f}, " + \
            f"Val_loss:{log['val_loss_list'][-1]:.3f}, " + \
            f"Min_loss:{min_loss:.4f} in epoch:{best_epoch}, " + \
            f"Lr:{learning_rates}"
        )
        lr_scheduler.step() # ExponentialLR
        # lr_scheduler.step(log['val_loss_list'][-1]) # ReduceLROnPlateau

    return min_loss, best_epoch, log

def main(args):
    start_time = time_synchronized()
    cfg = loadConfig(args.config)
    modeDir, ckptDir = makeDirectory('train', name=cfg['PROJECT_NAME'])
    cfg['PROJECT_NAME'] = os.path.split(modeDir)[-1]
    setupLogger(modeDir, mode='train')
    logging.info("Start training model")
    logConfig(cfg)
    with open(os.path.join(modeDir, f"{cfg['PROJECT_NAME']}.yaml"), 'w') as cfg_f:
        yaml.dump(cfg, cfg_f)
        
    # Define device
    if torch.cuda.is_available() and bool(cfg['GPUS']):
        torch.backends.cudnn.enabled = cfg['CUDNN']['ENABLED']
        torch.backends.cudnn.benchmark = cfg['CUDNN']['BENCHMARK']
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Define model
    model = vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1) # /home/vcpuser/.cache/torch/hub/checkpoints/vit_h_14_swag-80465313.pth
    # Freeze model parameter
    for param in model.parameters():
        param.requires_grad = cfg['TRAIN']['RESUME']
    # Define new output layer
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, cfg['DATASET']['NUM_CLASSES'])
    model = nn.DataParallel(model, device_ids=cfg['GPUS'])
    # Initialize model and data
    if cfg['TRAIN']['RESUME']:
        state_dict = torch.load(cfg['TRAIN']['PRETRAINED_MODEL'], map_location='cpu')
        model.load_state_dict(state_dict['model_dict'])
        log = {
            'train_loss_list' : state_dict['train_loss_list'],
            'train_acc_list' : state_dict['train_acc_list'],
            'val_loss_list' : state_dict['val_loss_list'],
            'val_acc_list' : state_dict['val_acc_list'],
        }
        min_loss = min(log['val_loss_list'])
        best_epoch = log['val_loss_list'].index(min_loss)
        last_epoch = len(log['val_acc_list'])
    else:
        log = {
            'train_loss_list':[],
            'train_acc_list':[],
            'val_loss_list':[],
            'val_acc_list':[],
        }
        min_loss = np.inf
        best_epoch = last_epoch = 0

    # Define dataloader
    train_datapath = os.path.join(cfg['DATASET']['ROOT'], cfg['DATASET']['TRAIN_SET'])
    x_train, y_train, x_val, y_val = loadDataset(cfg['DATASET']['NAME'], train_datapath, val_factor=0.1)
    dataset = MyDataset(x_train, y_train, data_aug=cfg['TRAIN']['DATA_AUG'], img_size=cfg['TRAIN']['IMAGE_SIZE'])
    val_dataset = MyDataset(x_val, y_val, data_aug=cfg['TEST']['DATA_AUG'], img_size=cfg['TEST']['IMAGE_SIZE'])

    scaler = GradScaler()
    for lr, nEpochs, batch_size_per_gpu in \
        zip(cfg['TRAIN']['LR'], cfg['TRAIN']['EPOCH'], cfg['TRAIN']['BATCH_SIZE_PER_GPU']):
        torch.cuda.empty_cache()
        batch_size = batch_size_per_gpu * len(cfg['GPUS'])
        val_batch_size = max(int(batch_size * 0.2), len(cfg['GPUS']), 1)

        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=cfg['TRAIN']['SHUFFLE'],
            num_workers=cfg['DATASET']['WORKERS'],
            pin_memory=True
            )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=cfg['TEST']['SHUFFLE'],
            num_workers=cfg['DATASET']['WORKERS'],
            pin_memory=True
            )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        min_loss, best_epoch, log = train(
            model,
            optimizer,
            lr_scheduler=ExponentialLR(optimizer, gamma=0.5),#ReduceLROnPlateau(optimizer, factor=0.8, patience=1),
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            min_loss=min_loss,
            best_epoch=best_epoch,
            log=log,
            scaler=GradScaler(),
            start_epochs=last_epoch,
            nEpochs=last_epoch + nEpochs,
            ckptDir=ckptDir,
            device=device,
            mixed_precision=cfg['MIXED_PRECISION']
            )
        last_epoch += nEpochs
            
        for param in model.parameters():
            param.requires_grad = True
        
    logging.info(time.strftime('Start: %Y.%m.%d %H:%M:%S',time.localtime(start_time)))
    logging.info(time.strftime('End: %Y.%m.%d %H:%M:%S',time.localtime(time_synchronized())))
    
    # Visualize accuracy and loss
    savePlot(log, os.path.join(modeDir, 'plot.png'))

if __name__=="__main__":
    args = parse_args()
    try:
        main(args)
    except Exception as e:
        logging.critical(traceback.format_exc())
