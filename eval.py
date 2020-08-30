import yaml
import argparse
import numpy as np
import os

from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
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


tt_logger = TestTubeLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    debug=False,
    create_git_tag=False,
    version=config['logging_params']['version'],
)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

model = vae_models[config['model_params']['name']](imsize=config['exp_params']['img_size'], **config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'])
model_save_path = '{}/{}/version_{}'.format(config['logging_params']['save_dir'], config['logging_params']['name'], tt_logger.version)

if config['logging_params']['resume'] ==None:
    weights = [x for x in os.listdir(model_save_path) if '.ckpt' in x]
    weights.sort(key=lambda x: os.path.getmtime(x))
    model_path = os.path.join(model_save_path,weights[0])
else:
    model_path = '{}/{}'.format(model_save_path, config['logging_params']['resume'])
experiment = VAEXperiment.load_from_checkpoint(model_path, vae_model = model, params=config['exp_params'])
experiment.eval()
experiment.freeze()
experiment.sample_interpolate(save_dir=config['logging_params']['save_dir'], name=config['logging_params']['name'],
                              version=config['logging_params']['version'], save_svg=True, other_interpolations=config['logging_params']['other_interpolations'])
