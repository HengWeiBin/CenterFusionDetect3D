from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pathlib import Path
import time
import logging
import torch

def createLogger(cfg, phase='train'):
    '''
    This function will make output directory and creates a logger.
    They will be returned.

    input:
    cfg: yacs config object
    phase: train or test

    output:
    logger: logger object
    final_output_dir: output directory
    '''
    root_output_dir = Path('output')
    if not root_output_dir.exists():
        print(f'=> creating {root_output_dir}')
        root_output_dir.mkdir()

    final_output_dir = root_output_dir / cfg.NAME
    if not final_output_dir.exists():
        print(f'=> creating {final_output_dir}')
        final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_dir = final_output_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f'{cfg.NAME}_{time_str}_{phase}.log'
    log_head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(log_file), format=log_head)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Add StreamHandler allow print to console when calling info()
    console = logging.StreamHandler()
    logger.addHandler(console)

    return logger, str(final_output_dir)

def saveModel(log, model, output_file):
    '''
    Save model and log to output_file
    '''
    log['model_dict'] = model.state_dict()
    torch.save(log, output_file)