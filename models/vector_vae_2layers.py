import random

from models import VectorVAE
import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class VectorVAE2Layers(VectorVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 loss_fn: str = 'MSE',
                 imsize: int = 128,
                 paths: int = 4,
                 **kwargs) -> None:
        super(VectorVAE2Layers, self).__init__(in_channels,
                 latent_dim,
                 hidden_dims,
                 loss_fn,
                 imsize,
                 paths,
                 **kwargs)
        def get_computational_unit(in_channels, out_channels, unit):
            if unit=='conv':
                return nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2, padding_mode='circular', stride=1, dilation=1)
            else:
                return nn.Linear(in_channels, out_channels)
        self.colors = [[0, 0, 0, 1], [255/255, 165/255, 0/255, 1],]
        self.divide_shape = nn.Sequential(
            get_computational_unit(latent_dim+2, latent_dim, 'mlp'),
            nn.ReLU(),  # bound spatial extent
            get_computational_unit(latent_dim, latent_dim, 'mlp'),
            nn.ReLU(),  # bound spatial extent
            get_computational_unit(latent_dim, latent_dim, 'mlp'),
            nn.ReLU()  # bound spatial extent
        )
        layer_id = torch.tensor([[1,0],[0,1]], dtype=torch.float32)
        self.register_buffer('layer_id', layer_id)


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        output = self.decode_and_composite(z, verbose=False)
        return [output, input, mu, log_var]


    def decode_and_composite(self, z: Tensor, **kwargs):
        bs = z.shape[0]
        layers = []
        for i in range(2):
            layer_id_repeat = self.layer_id[i:i+1].repeat([bs, 1])
            z_id = torch.cat([z, layer_id_repeat], dim=1)
            shape_latent = self.divide_shape(z_id)
            all_points = self.decode(shape_latent)
            layer = self.raster(all_points, self.colors[i], verbose=kwargs['verbose'] )
            layers.append(layer)
        output = layers[0][:, :3]*(1-layers[1][:,3:4,:,:]) + layers[1][:, :3]*layers[1][:,3:4,:,:]
        return output


    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        output = self.decode_and_composite(z, verbose=random.choice([True, False]))
        return  output#[:, :3]


    def interpolate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        all_interpolations = []
        for i in range(mu.shape[0]):
            z = self.interpolate_vectors(mu[2], mu[i], 10)
            output = self.decode_and_composite(z, verbose=kwargs['verbose'])
            all_interpolations.append(output)
        return all_interpolations

    def interpolate2D(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        all_interpolations = []
        y_axis = self.interpolate_vectors(mu[12], mu[6], 10)
        for i in range(10):
            z = self.interpolate_vectors(y_axis[i], mu[9], 10)
            output = self.decode_and_composite(z, verbose=kwargs['verbose'])
            all_interpolations.append(output)
        return all_interpolations


    def naive_vector_interpolate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        all_interpolations = []
        bs = mu.shape[0]
        for j in range(mu.shape[0]):
            layers = []
            for i in range(2):
                layer_id_repeat = self.layer_id[i:i+1].repeat([bs, 1])
                z_id = torch.cat([mu, layer_id_repeat], dim=1)
                shape_latent = self.divide_shape(z_id)
                all_points = self.decode(shape_latent)
                all_points_interpolate = self.interpolate_vectors(all_points[2], all_points[j], 10)
                layer = self.raster(all_points_interpolate, self.colors[i], verbose=kwargs['verbose'] )
                layers.append(layer)
            output = layers[0][:, :3]*(1-layers[1][:,3:4,:,:]) + layers[1][:, :3]*layers[1][:,3:4,:,:]
            all_interpolations.append(output)
        return all_interpolations


    def visualize_sampling(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        all_interpolations = []
        for i in range(7,25):
            self.redo_features(i)
            output = self.decode_and_composite(mu, verbose=kwargs['verbose'])
            all_interpolations.append(output)
        return all_interpolations