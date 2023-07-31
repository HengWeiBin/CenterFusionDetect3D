import os
from pathlib import Path

def createLogger(cfg, phase='train'):
    '''
    This function will make output directory and creates a logger.
    Logger will be returned.

    input:
    cfg: yacs config object
    phase: train or test
    '''
    root_output_dir = Path(os.path.join('..', 'output'))
    if not root_output_dir.exists():
        print(f'=> creating {root_output_dir}')
        root_output_dir.mkdir()