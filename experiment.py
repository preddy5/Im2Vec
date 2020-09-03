import itertools
import math
import os
import random

import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from torchvision import datasets
from scheduler import GradualWarmupScheduler
import matplotlib.pyplot as plt


class ImageFileDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        path, _ = self.samples[index]
        dirs, filename = os.path.split(path)
        _, class_name = os.path.split(dirs)
        filename = os.path.join(class_name, filename)
        return sample


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['batch_size']/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.logger.experiment.log({'loss': train_loss['loss']})
        if self.params['grow']:
            self.model.redo_features(random.randint(7, 25))
        return train_loss

    # def validation_step(self, batch, batch_idx, optimizer_idx = 0):
    #     return
    #     real_img, labels = batch
    #     self.curr_device = real_img.device
    #
    #     results = self.forward(real_img, labels = labels)
    #     val_loss = self.model.loss_function(*results,
    #                                         M_N = self.params['batch_size']/ self.num_val_imgs,
    #                                         optimizer_idx = optimizer_idx,
    #                                         batch_idx = batch_idx)
    #
    #     return val_loss

    def training_epoch_end(self, outputs):
        super(VAEXperiment, self).training_epoch_end(outputs)
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        if (self.current_epoch) % 50 == 0:
            self.model.beta = min(self.model.beta*2, 1000)
            print(self.model.beta)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}
    #
    # def on_after_backward(self):
    #     # example to inspect gradient information in tensorboard
    #     if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
    #         params = self.state_dict()
    #         for k, v in params.items():
    #             if k == 'model.point_predictor.11.weight':
    #                 grads = v
    #                 name = k
    #                 self.logger.experiment.add_histogram(tag=name, values=grads,
    #                                                      global_step=self.trainer.global_step)

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(recons.data,
                          f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          f"recons_{self.logger.name}_{self.current_epoch:04d}.png",
                          normalize=True,
                          nrow=12)

        vutils.save_image(test_input.data,
                          f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          f"real_img_{self.logger.name}_{self.current_epoch:04d}.png",
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels = test_label)
            vutils.save_image(samples.cpu().data,
                              f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                              f"{self.logger.name}_{self.current_epoch:04d}.png",
                              normalize=True,
                              nrow=12)

        except:
            pass


        del test_input, recons #, samples

    def sample_interpolate(self, save_dir, name, version, save_svg=False, other_interpolations=False):
        train_sampler = self.train_dataloader()
        train_input, train_label = next(iter(train_sampler))
        vutils.save_image(train_input.cpu().data,
                          f"{save_dir}{name}/"
                          f"{name}_input.png",
                          normalize=False,
                          nrow=16)
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        interpolate_samples = self.model.interpolate(test_input, verbose=False)
        interpolate_samples = torch.cat(interpolate_samples, dim=0)
        vutils.save_image(interpolate_samples.cpu().data,
                          f"{save_dir}{name}/version_{version}/"
                          f"{name}_interpolate_img.png",
                          normalize=False,
                          nrow=10)

        if other_interpolations:
            self.model.sampling_error(test_input)
            interpolate_samples = self.model.interpolate2D(test_input, verbose=False)
            interpolate_samples = torch.cat(interpolate_samples, dim=0)
            vutils.save_image(interpolate_samples.cpu().data,
                              f"{save_dir}{name}/version_{version}/"
                              f"{name}_interpolate2D_image.png",
                              normalize=True,
                              nrow=10)
            interpolate_samples = self.model.interpolate2D(test_input, verbose=True)
            interpolate_samples = torch.cat(interpolate_samples, dim=0)
            vutils.save_image(interpolate_samples.cpu().data,
                              f"{save_dir}{name}/version_{version}/"
                              f"{name}_interpolate2D_vector.png",
                              normalize=True,
                              nrow=10)
            interpolate_samples = self.model.visualize_sampling(test_input, verbose=False)
            interpolate_samples = torch.cat(interpolate_samples, dim=0)
            vutils.save_image(interpolate_samples.cpu().data,
                              f"{save_dir}{name}/version_{version}/"
                              f"{name}_visualize_sampling_image.png",
                              normalize=True,
                              nrow=self.params['val_batch_size'])
            interpolate_samples = self.model.visualize_sampling(test_input, verbose=True)
            interpolate_samples = torch.cat(interpolate_samples, dim=0)
            vutils.save_image(interpolate_samples.cpu().data,
                              f"{save_dir}{name}/version_{version}/"
                              f"{name}_visualize_sampling_vector.png",
                              normalize=True,
                              nrow=self.params['val_batch_size'])
            interpolate_samples = self.model.naive_vector_interpolate(test_input, verbose=False)
            interpolate_samples = torch.cat(interpolate_samples, dim=0)
            vutils.save_image(interpolate_samples.cpu().data,
                              f"{save_dir}{name}/version_{version}/"
                              f"{name}_naive_interpolate_image.png",
                              normalize=True,
                              nrow=10)
            interpolate_samples = self.model.naive_vector_interpolate(test_input, verbose=True)
            interpolate_samples = torch.cat(interpolate_samples, dim=0)
            vutils.save_image(interpolate_samples.cpu().data,
                              f"{save_dir}{name}/version_{version}/"
                              f"{name}_naive_interpolate_vector.png",
                              normalize=True,
                              nrow=10)
            interpolate_samples = self.model.interpolate(test_input, verbose=True)
            interpolate_samples = torch.cat(interpolate_samples, dim=0)
            vutils.save_image(interpolate_samples.cpu().data,
                              f"{save_dir}{name}/version_{version}/"
                              f"{name}_interpolate_vector.png",
                              normalize=True,
                              nrow=10)
            if self.model.only_auxillary_training:
                graph = self.model.visualize_aux_error(test_input, verbose=True)
                plt.imsave(f"{save_dir}{name}/version_{version}/{name}_aux_graph.png", graph)

        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(recons.cpu().data,
                          f"{save_dir}{name}/version_{version}/"
                          f"{name}_recons.png",
                          normalize=True,
                          nrow=16)
        vutils.save_image(test_input.cpu().data,
                          f"{save_dir}{name}/version_{version}/"
                          f"{name}_input.png",
                          normalize=True,
                          nrow=10)
        # if save_svg:
        #     self.model.save(test_input, save_dir, name)

    def configure_optimizers(self):

        optims = []
        scheds = []
        if self.model.only_auxillary_training:
            print('Learning Rate changed for auxillary training')
            self.params['LR'] = 0.00001
        optimizer = optim.Adam(self.model.parameters(),
                                   lr=self.params['LR'],
                                   weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1, total_epoch=5,
                                                          after_scheduler=scheduler)

                scheds.append(scheduler_warmup)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    @data_loader
    def train_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            dataset = CelebA(root = self.params['data_path'],
                             split = "train",
                             transform=transform,
                             download=False)
        elif self.params['dataset'] == 'MNIST':
            dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            self.sample_dataloader =  DataLoader(dataset,
                                                 batch_size= 64,
                                                 shuffle = False,
                                                 drop_last=True)
            self.num_val_imgs = 200
            in_channels = 1
        else:
            dataset = datasets.ImageFolder(self.params['data_path'], transform=transform)
            self.sample_dataloader =  DataLoader(dataset,
                                                 batch_size= self.params['val_batch_size'],
                                                 shuffle = self.params['val_shuffle'],
                                                 drop_last=True, num_workers=1)
            self.num_val_imgs = len(self.sample_dataloader)

            # raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size= self.params['batch_size'],
                          shuffle = True,
                          drop_last=False,
                          num_workers=4)

    # @data_loader
    # def val_dataloader(self):
    #     transform = self.data_transforms()
    #
    #     if self.params['dataset'] == 'celeba':
    #         self.sample_dataloader =  DataLoader(CelebA(root = self.params['data_path'],
    #                                                     split = "test",
    #                                                     transform=transform,
    #                                                     download=False),
    #                                              batch_size= 144,
    #                                              shuffle = True,
    #                                              drop_last=True)
    #         self.num_val_imgs = len(self.sample_dataloader)
    #     else:
    #         dataset = datasets.ImageFolder(self.params['data_path'], transform=transform)
    #         self.sample_dataloader =  DataLoader(dataset,
    #                                              batch_size= 64,
    #                                              shuffle = False,
    #                                              drop_last=True)
    #         self.num_val_imgs = 200#len(self.sample_dataloader)
    #
    #     return self.sample_dataloader
    #
    def data_transforms(self):

        SetRange = transforms.Lambda(lambda X: (2 * X - 1.))
        SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))

        if self.params['dataset'] == 'celeba':
            transform = transforms.Compose([#transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])
        else:
            transform = transforms.Compose([#transforms.RandomHorizontalFlip(),
                                            transforms.Resize(self.params['img_size']),
                                            # transforms.RandomRotation([0, 360], resample=3, fill=(255,255,255)),
                                            # transforms.RandomAffine([0, 0], (0.0,0.05), (1.0,1.0), resample=3, fillcolor=(255,255,255)),
                                            transforms.CenterCrop(self.params['img_size']),
                                            transforms.ToTensor(),
                                            ])
            # raise ValueError('Undefined dataset type')
        return transform
