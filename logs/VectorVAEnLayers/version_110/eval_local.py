import yaml
import argparse
import numpy as np
import os

from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

model_save_path = os.getcwd()#'{}/{}/version_{}'.format(config['logging_params']['save_dir'], config['logging_params']['name'], tt_logger.version)
parent = '/'.join(model_save_path.split('/')[:-3])
config['logging_params']['save_dir'] = os.path.join(parent, config['logging_params']['save_dir'])
config['exp_params']['data_path'] = os.path.join(parent, config['exp_params']['data_path'])
print(parent, config['exp_params']['data_path'])

model = vae_models[config['model_params']['name']](imsize=config['exp_params']['img_size'], **config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'])

weights = [x for x in os.listdir(model_save_path) if '.ckpt' in x]
weights.sort(key=lambda x: os.path.getmtime(x))
load_weight = weights[-1]
print('loading: ', load_weight)

checkpoint = torch.load(load_weight)
experiment.load_state_dict(checkpoint['state_dict'])
_ = experiment.train_dataloader()
experiment.eval()
experiment.freeze()
experiment.sample_interpolate(save_dir=config['logging_params']['save_dir'], name=config['logging_params']['name'],
                              version=config['logging_params']['version'], save_svg=True, other_interpolations=config['logging_params']['other_interpolations'])
