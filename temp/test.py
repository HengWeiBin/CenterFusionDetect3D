import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import time
import numpy as np
from dataset import MyDataset
from tqdm import tqdm
import logging
import argparse
from utils import makeDirectory, setupLogger, loadDataset, logConfig, loadConfig, time_synchronized
from torchvision.models import vit_h_14, ViT_H_14_Weights
import yaml
import traceback

def parse_args():
    # python test.py --config config.yaml
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--config', help='Configuration file', default='config.yaml', type=str)
    return parser.parse_args()

def main(args):
    start_time = time_synchronized()
    cfg = loadConfig(args.config)
    modeDir = makeDirectory('test', name=cfg['PROJECT_NAME'])
    setupLogger(modeDir, mode='test')
    logging.info("Start testing ViT model")
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
    test_datapath = os.path.join(cfg['DATASET']['ROOT'], cfg['DATASET']['TEST_SET'])
    test_images = loadDataset(cfg['DATASET']['NAME'], test_datapath, test=True)
    dataset = MyDataset(test_images, img_size=cfg['TEST']['IMAGE_SIZE'], data_aug=cfg['TEST']['DATA_AUG'])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg['TEST']['BATCH_SIZE_PER_GPU'],
        shuffle=cfg['TEST']['SHUFFLE'],
        num_workers=cfg['DATASET']['WORKERS'],
        pin_memory=True
        )

    with torch.no_grad():
        # Inference
        model.eval()
        pbar = tqdm(dataloader, desc='Inference')
        with open(os.path.join(modeDir, 'pred_results.csv'), 'w') as f:
            f.write('Id,Category\n')
            current_id = 0
            for i, img in enumerate(pbar):
                img = img.to(device, non_blocking=True)
                img = img.half() if cfg['MIXED_PRECISION'] else img.float()
                with autocast(enabled=cfg['MIXED_PRECISION']):
                    # predict
                    t1 = time_synchronized()
                    predict = model(img)
                pred_cls = torch.argmax(predict, dim=1)
                for single_pred in pred_cls:
                    f.write(f'{current_id},{single_pred}\n')
                    current_id += 1
                pbar_msg = f"Inference {1e3 * (time_synchronized() - t1) / len(img):.2f}ms/img"
                pbar.set_description(pbar_msg)

    logging.info(time.strftime('Start: %Y.%m.%d %H:%M:%S',time.localtime(start_time)))
    logging.info(time.strftime('End: %Y.%m.%d %H:%M:%S',time.localtime(time_synchronized())))

if __name__=="__main__":
    args = parse_args()
    try:
        main(args)
    except Exception as e:
        logging.critical(traceback.format_exc())