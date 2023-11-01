import argparse
import collections
import numpy as np
#import os
import torch
from data_loader import data_loaders
from trainer import trainers
from torchvision.utils import save_image
from torch.optim import Adam
from parse_config import ConfigParser

# change workspace
#os.chdir('.\DiffGenMol')

# fix random seeds for reproducibility
SEED = 2807
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', data_loaders, config=config)
    logger.info(data_loader)
    logger.info(f'seq_length: {data_loader.get_seq_length()}')
    
    trainer = config.init_obj('trainer', trainers, config=config, data_loader=data_loader)
    logger.info(trainer)

    trainer.train()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='DiffGenMol')
    args.add_argument('-c', '--config', default="config_selfies_logp_light_qm9.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='trainer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
