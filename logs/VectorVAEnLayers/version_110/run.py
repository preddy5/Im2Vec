
from shutil import copytree, ignore_patterns, rmtree
import json

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
import click

rootdir = os.getcwd()
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
resume = False
model_save_path = '{}/{}/version_{}/'.format(config['logging_params']['save_dir'], config['logging_params']['name'], tt_logger.version)
print(model_save_path)
# Copying the folder
if os.path.exists(model_save_path):
    if config['model_params']['only_auxillary_training'] or config['model_params']['memory_leak_training']:
        print('Training Auxillary Network or Memory Leak')
    elif click.confirm('Folder exists do you want to override?', default=True):
        rmtree(model_save_path)
        copytree(rootdir, model_save_path, ignore=ignore_patterns('*.pyc', 'tmp*', 'logs*', 'data*'))
    else:
        resume = True
else:
    copytree(rootdir, model_save_path, ignore=ignore_patterns('*.pyc', 'tmp*', 'logs*', 'data*'))

with open(model_save_path+'hyperparameters.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)


# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False
print(config['model_params'])
model = vae_models[config['model_params']['name']](imsize=config['exp_params']['img_size'], **config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'])

model_path = None
if config['model_params']['only_auxillary_training'] or config['model_params']['memory_leak_training'] or resume:
    weights = [os.path.join(model_save_path, x) for x in os.listdir(model_save_path) if '.ckpt' in x]
    weights.sort(key=lambda x: os.path.getmtime(x))
    if len(weights)>0:
        model_path = weights[-1]
        print('loading: ', weights[-1])
        if config['model_params']['only_auxillary_training']:
            checkpoint = torch.load(model_path)
            experiment.load_state_dict(checkpoint['state_dict'])
            model_path = None
    # experiment = VAEXperiment.load_from_checkpoint(model_path, vae_model = model, params=config['exp_params'])

checkpoint_callback = ModelCheckpoint(model_save_path,
                                      verbose=True, save_last=True)
                                      # monitor='loss',
                                      # mode='min',)
                                      # save_top_k=5,)

print(config['exp_params'], config['logging_params']['save_dir']+config['logging_params']['name'])
runner = Trainer(checkpoint_callback=checkpoint_callback,
                 resume_from_checkpoint=model_path,
                 logger=tt_logger,
                 log_save_interval=100,
                 # gradient_clip_val=0.5,
                 # train_percent_check=1.,
                 # val_percent_check=1.,
                 # num_sanity_val_steps=1,
                 weights_summary='full',
                 early_stop_callback = False,
                 **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment)
# experiment.train_dataloader()
# experiment.sample_interpolate()
