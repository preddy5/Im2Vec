import random

from models import VectorVAE
import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class VectorVAEnLayers(VectorVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 loss_fn: str = 'MSE',
                 imsize: int = 128,
                 paths: int = 4,
                 **kwargs) -> None:
        super(VectorVAEnLayers, self).__init__(in_channels,
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
        self.colors = [[0, 0, 0, 1], [255/255, 165/255, 0/255, 1], [0/255, 0/255, 255/255, 1],]

        self.rnn = nn.LSTM(latent_dim, latent_dim, 2, bidirectional=True)

        self.divide_shape = nn.Sequential(
            get_computational_unit(latent_dim, latent_dim, 'mlp'),
            nn.ReLU(),  # bound spatial extent
            get_computational_unit(latent_dim, latent_dim, 'mlp'),
            nn.ReLU(),  # bound spatial extent
            get_computational_unit(latent_dim, latent_dim, 'mlp'),
            nn.ReLU()  # bound spatial extent
        )
        self.z_order = nn.Sequential(
            get_computational_unit(latent_dim, latent_dim, 'mlp'),
            nn.ReLU(),  # bound spatial extent
            get_computational_unit(latent_dim, latent_dim, 'mlp'),
            nn.ReLU(),  # bound spatial extent
            get_computational_unit(latent_dim, 1, 'mlp'),
        )
        layer_id = torch.tensor([[1,0,0],[0,1,0],[0,0,1]], dtype=torch.float32)
        self.register_buffer('layer_id', layer_id)


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        output = self.decode_and_composite(z, verbose=False, return_overlap_loss=False)
        return [output, input, mu, log_var, 0]


    def decode_and_composite(self, z: Tensor, return_overlap_loss=False, **kwargs):
        bs = z.shape[0]
        layers = []
        n = 3
        z_rnn_input = z[None, :, :].repeat(n, 1, 1)  # [len, batch size, emb dim]
        outputs, hidden = self.rnn(z_rnn_input)
        outputs = outputs.permute(1, 0, 2)  # [batch size, len, emb dim]
        outputs = outputs[:, :, :self.latent_dim] + outputs[:, :, self.latent_dim:]
        z_s = []
        for i in range(n):
            shape_latent = self.divide_shape(outputs[:, i, :])
            all_points = self.decode(shape_latent)
            layer = self.raster(all_points, self.colors[i], verbose=kwargs['verbose'], white_background=False )
            z = self.z_order(shape_latent)
            layers.append(layer)
            z_s.append(torch.exp(z))
        # output = (layers[0][:, :3] * layers[0][:, 3:4, :, :] * (1 - layers[1][:, 3:4, :, :]) * (
        #             1 - layers[2][:, 3:4, :, :])) + \
        #          (layers[1][:, :3] * layers[1][:, 3:4, :, :] * (1 - layers[2][:, 3:4, :, :])) + \
        #          (layers[2][:, :3] * layers[2][:, 3:4, :, :]) + \
        #          ((1 - layers[0][:, 3:4, :, :]) * (1 - layers[1][:, 3:4, :, :]) * (1 - layers[2][:, 3:4, :, :]))
        # inv_mask = ((1 - layers[0][:, 3:4, :, :]) * (1 - layers[1][:, 3:4, :, :]) * (1 - layers[2][:, 3:4, :, :]))
        # output = (layers[0][:, :3] * layers[0][:, 3:4, :, :] * (1 - layers[1][:, 3:4, :, :]) * (
        #             1 - layers[2][:, 3:4, :, :])) + \
        #          (layers[1][:, :3] * layers[1][:, 3:4, :, :]) + \
        #          (layers[2][:, :3] * layers[2][:, 3:4, :, :])
        # output = output * (1-inv_mask) + inv_mask


        inv_mask = ((1 - layers[0][:, 3:4, :, :]) * (1 - layers[1][:, 3:4, :, :]) * (1 - layers[2][:, 3:4, :, :]))
        sum_alpha = layers[0][:, 3:4, :, :]*z[0] + layers[1][:, 3:4, :, :]*z[1] + layers[2][:, 3:4, :, :]*z[2] + inv_mask
        alpha0 = layers[0][:, 3:4, :, :]*z[0] / sum_alpha
        alpha1 = layers[1][:, 3:4, :, :]*z[1] / sum_alpha
        alpha2 = layers[2][:, 3:4, :, :]*z[2] / sum_alpha
        inv_mask = inv_mask/sum_alpha
        output = (layers[0][:, :3] * alpha0) + \
                 (layers[1][:, :3] * alpha1) + \
                 (layers[2][:, :3] * alpha2)
        output = output * (1-inv_mask) + inv_mask
        if return_overlap_loss:
            overlap_alpha = layers[1][:,3:4,:,:] + layers[2][:,3:4,:,:]
            loss = F.relu(overlap_alpha - 1).mean()
            return output, loss
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
        n = 3
        for j in range(bs):
            layers = []
            z_rnn_input = mu[None, :, :].repeat(n, 1, 1)  # [len, batch size, emb dim]
            outputs, hidden = self.rnn(z_rnn_input)
            outputs = outputs.permute(1, 0, 2)  # [batch size, len, emb dim]
            outputs = outputs[:, :, :self.latent_dim] + outputs[:, :, self.latent_dim:]
            for i in range(n):
                shape_latent = self.divide_shape(outputs[:, i, :])
                all_points = self.decode(shape_latent)
                all_points_interpolate = self.interpolate_vectors(all_points[2], all_points[j], 10)
                layer = self.raster(all_points_interpolate, self.colors[i], verbose=kwargs['verbose'] )
                layers.append(layer)
            # output = (layers[0][:, :3] * layers[0][:, 3:4, :, :] * (1 - layers[1][:, 3:4, :, :]) * (
            #             1 - layers[2][:, 3:4, :, :])) + \
            #          (layers[1][:, :3] * layers[1][:, 3:4, :, :] * (1 - layers[2][:, 3:4, :, :])) + \
            #          (layers[2][:, :3] * layers[2][:, 3:4, :, :]) + \
            #          ((1 - layers[0][:, 3:4, :, :]) * (1 - layers[1][:, 3:4, :, :]) * (1 - layers[2][:, 3:4, :, :]))
            # output = (layers[0][:, :3] * layers[0][:, 3:4, :, :] * (1 - layers[1][:, 3:4, :, :]) * (
            #             1 - layers[2][:, 3:4, :, :])) + \
            #          (layers[1][:, :3] * layers[1][:, 3:4, :, :]) + \
            #          (layers[2][:, :3] * layers[2][:, 3:4, :, :]) + \
            #          ((1 - layers[0][:, 3:4, :, :]) * (1 - layers[1][:, 3:4, :, :]) * (1 - layers[2][:, 3:4, :, :]))

            sum_alpha = layers[1][:, 3:4, :, :] + layers[2][:, 3:4, :, :]
            alpha1 = layers[1][:, 3:4, :, :]/sum_alpha
            alpha2 = layers[2][:, 3:4, :, :]/sum_alpha
            mask = ((1 - layers[0][:, 3:4, :, :]) * (1 - alpha1) * (1 - alpha2))
            output = (layers[0][:, :3] * layers[0][:, 3:4, :, :] * (1 - layers[1][:, 3:4, :, :]) * (
                        1 - layers[2][:, 3:4, :, :])) + \
                     (layers[1][:, :3] * alpha1) + \
                     (layers[2][:, :3] * alpha2)
            output = output*mask + (1-mask)
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