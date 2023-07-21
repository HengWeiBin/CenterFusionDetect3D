import numpy as np
import torch
import logging
import os
import time
import random
import matplotlib.pyplot as plt
import yaml

def loadDataset(name, img_root, val_factor=0.1, test=False):
    if name == 'NtutEMnist':
        return loadNtutEMNIST(img_root, val_factor, test)
    elif name == 'TWFood101':
        return loadTwFood(img_root, val_factor, test)

def loadTwFood(csv_root, val_factor=0, test=False):
    datasetDirHead, _ = os.path.split(csv_root)

    with open(csv_root, 'r') as f:
        lines = f.readlines()

    images = []
    labels = []
    for line in lines:
        if test:
            _, path = line.split(',')
        else:
            _, label, path = line.split(',')
            labels.append(int(label))
        path = os.path.join(datasetDirHead, path[:-1])
        images.append(path)
    
    if test:
        return images

    dataset = [*zip(images, labels)]
    random.shuffle(dataset)
    images, labels = zip(*dataset)
    images = np.array(images)
    labels = np.array(labels)

    val_size = int(len(images) * val_factor)
    x_val = images[:val_size]
    y_val = labels[:val_size]
    x_train = images[val_size:]
    y_train = labels[val_size:]
    return x_train, y_train, x_val, y_val    

def loadNtutEMNIST(img_root, val_factor=0, test=False, mean=0.1736, std=0.3246):
    data = np.load(img_root)
    if test:
        images = data['testing_images']
    else:
        images = data['training_images']
        labels = data['training_labels']

    images = images.reshape((images.shape[0], 28, 28, 1))
    images = (images - mean) / std

    if test:
        return images

    val_size = int(images.shape[0] * val_factor)
    x_val = images[:val_size]
    y_val = labels[:val_size]
    x_train = images[val_size:]
    y_train = labels[val_size:]

    return x_train, y_train, x_val, y_val

def saveModel(model, log:dict, save_path:str='checkpoint.pt') -> None:
    '''
    Save pytorch model as state_dict
    input:
        model: pytorch model
        log: a dictionary with training information
            {
                'loss_train_log': list[],
                'loss_val_log': list[],
                ...
            }
        save_name: a path with filename of saved model
    '''
    log['model_dict'] = model.state_dict()
    torch.save(log, save_path)
    logging.info(f'Model saved to: {save_path}')

def getProjectDir(modeDir:str, name, n):
    dirName = name + str(n) if n else name
    if not os.path.exists(projectDir := os.path.join(modeDir, dirName)):
        return projectDir
    else:
        return getProjectDir(modeDir, name, n + 1)

def makeDirectory(mode:str, name:str=None):
    if not os.path.exists(outputDir := 'runs'):
        os.mkdir(outputDir)
    if not os.path.exists(modeDir := os.path.join(outputDir, mode)):
        os.mkdir(modeDir)
    name = mode if name is None else name
    projectDir = getProjectDir(modeDir, name, 0)
    os.mkdir(projectDir)
    
    if mode == 'train':
        if not os.path.exists(ckptDir := os.path.join(projectDir, 'checkpoints')):
            os.mkdir(ckptDir)
        return projectDir, ckptDir
    else:
        return projectDir

def setupLogger(savePath:str, mode:str):
    datetime = time.strftime('%Y_%m_%d-%H_%M_%S', time.localtime(time.time()))
    logging.basicConfig(
        filename=os.path.join(savePath, f'{mode}_log_{datetime}.log'),
        level=logging.INFO, 
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p'
    )
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

def savePlot(log, save_path):
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.title("Loss")
    plt.plot(log['val_loss_list'], label="val")
    plt.plot(log['train_loss_list'], label="train")
    plt.legend()
    plt.subplot(122)
    plt.title("Accuracy")
    plt.plot(log['val_acc_list'], label="val")
    plt.plot(log['train_acc_list'], label="train")
    plt.legend()
    plt.savefig(save_path)

def logConfig(config, prefix:str=''):
    for key, value in config.items():
        if isinstance(value, dict):
            logging.info(f'{prefix}{key}:')
            logConfig(value, prefix='  ')
        else:
            logging.info(f'{prefix}{key}:{value}')

def loadConfig(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

if __name__ == '__main__':
    config = loadConfig('config.yaml')
    print(config)