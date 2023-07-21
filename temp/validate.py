import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import time
import numpy as np
from dataset import MyDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import makeDirectory, setupLogger, loadDataset, logConfig, loadConfig, time_synchronized
import logging
import sys
import yaml
import traceback

def parse_args():
    # python validate.py --config config.yaml
    parser = argparse.ArgumentParser(description='Validation')
    parser.add_argument('--config', help='Configuration file', default='config.yaml', type=str)
    return parser.parse_args()

def main(args):
    start_time = time_synchronized()
    cfg = loadConfig(args.config)
    modeDir = makeDirectory('validate', name=cfg['PROJECT_NAME'])
    setupLogger(modeDir, 'validate')
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
    # Define new output layer
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, 101)
    model = nn.DataParallel(model, device_ids=cfg['GPUS']).to(device)
    # Load weight
    model_param_path = os.path.join('runs', 'train', cfg['PROJECT_NAME'], 'checkpoints', 'best.pt')
    state_dict = torch.load(model_param_path, map_location='cpu')
    model.load_state_dict(state_dict['model_dict'])

    # Define dataloader
    train_datapath = os.path.join(cfg['DATASET']['ROOT'], cfg['DATASET']['TRAIN_SET'])
    _, _, x_val, y_val = loadDataset(cfg['DATASET']['NAME'], train_datapath, val_factor=0.1)

    val_dataset = MyDataset(x_val, y_val, img_size=cfg['TEST']['IMAGE_SIZE'], data_aug=cfg['TEST']['DATA_AUG'])
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg['TEST']['BATCH_SIZE_PER_GPU'],
        shuffle=cfg['TEST']['SHUFFLE'],
        num_workers=cfg['DATASET']['WORKERS'],
        pin_memory=True
        )
    n_val_batches = len(val_dataloader)

    # Validation
    with torch.no_grad():
        model.eval()
        accuracy = total_loss = 0
        for i,(img, label) in enumerate(tqdm(val_dataloader, desc='Validation')):
            img = img.to(device, non_blocking=True)
            img = img.half() if cfg['MIXED_PRECISION'] else img.float()
            label = label.to(device, non_blocking=True).flatten().long()
            with autocast(enabled=cfg['MIXED_PRECISION']):
                # forward propagation
                predict = model(img)
                # calculate loss
                loss = F.cross_entropy(predict, label)
            total_loss += loss.item()
            # calculate accuracy
            pred_one_hot = torch.argmax(predict, dim=1)
            accuracy += (pred_one_hot == label).float().mean()

    logging.info(
        f"Accuracy:{(accuracy / n_val_batches):.4f}, " + \
        f"Val_loss:{(total_loss / n_val_batches):.3f}"
    )
        
    logging.info(time.strftime('Start: %Y.%m.%d %H:%M:%S',time.localtime(start_time)))
    logging.info(time.strftime('End: %Y.%m.%d %H:%M:%S',time.localtime(time.time())))

if __name__=="__main__":
    args = parse_args()
    try:
        main(args)
    except Exception as e:
        logging.critical(traceback.format_exc())