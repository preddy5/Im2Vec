import random

import torch
from torch import nn
from torch.nn import functional as F

from models import VectorVAE
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
            if unit == 'conv':
                return nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2, padding_mode='circular', stride=1,
                                 dilation=1)
            else:
                return nn.Linear(in_channels, out_channels)

        # self.colors = [[0, 0, 0, 1], [255/255, 0/255, 0/255, 1],]
        self.colors = [[0, 0, 0, 1], [255/255, 0/255, 255/255, 1], [0/255, 255/255, 255/255, 1],]
        # self.colors = [[252 / 255, 194 / 255, 27 / 255, 1], [255 / 255, 0 / 255, 0 / 255, 1],
        #                [0 / 255, 255 / 255, 0 / 255, 1], [0 / 255, 0 / 255, 255 / 255, 1], ]
        # sample_rate = 1
        # n = len(self.colors)
        # del self.point_predictor
        # self.point_predictor = nn.ModuleList([])
        # num_one_hot = self.base_control_features.shape[1]
        # unit='conv'
        # fused_latent_dim = latent_dim + num_one_hot+ (sample_rate*2)
        # for i in range(n):
        #     self.point_predictor.append(nn.ModuleList([
        #     get_computational_unit(fused_latent_dim*2, fused_latent_dim*2, unit),
        #     get_computational_unit(fused_latent_dim*2, fused_latent_dim*2, unit),
        #     get_computational_unit(fused_latent_dim*2, fused_latent_dim*2, unit),
        #     get_computational_unit(fused_latent_dim*2, 2, unit),
        #     # nn.Sigmoid()  # bound spatial extent
        # ]))
        self.rnn = nn.LSTM(latent_dim, latent_dim, 2, bidirectional=True)
        self.composite_fn = self.hard_composite
        if kwargs['composite_fn'] == 'soft':
            print('Using Differential Compositing')
            self.composite_fn = self.soft_composite
        self.divide_shape = nn.Sequential(
            nn.ReLU(),  # bound spatial extent
            # get_computational_unit(latent_dim, latent_dim, 'mlp'),
            # nn.ReLU(),  # bound spatial extent
            # get_computational_unit(latent_dim, latent_dim, 'mlp'),
            # nn.ReLU(),  # bound spatial extent
        )
        self.final_shape_latent = nn.Sequential(
            get_computational_unit(latent_dim, latent_dim, 'mlp'),
            nn.ReLU(),  # bound spatial extent
            get_computational_unit(latent_dim, latent_dim, 'mlp'),
            nn.ReLU(),  # bound spatial extent
        )
        self.z_order = nn.Sequential(
            # get_computational_unit(latent_dim, latent_dim, 'mlp'),
            # nn.ReLU(),  # bound spatial extent
            # get_computational_unit(latent_dim, latent_dim, 'mlp'),
            # nn.ReLU(),  # bound spatial extent
            get_computational_unit(latent_dim, 1, 'mlp'),
        )
        layer_id = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        self.register_buffer('layer_id', layer_id)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        output, control_loss = self.decode_and_composite(z, verbose=False, return_overlap_loss=True)
        return [output, input, mu, log_var, control_loss]

    def hard_composite(self, **kwargs):
        layers = kwargs['layers']
        n = len(layers)
        alpha = (1 - layers[n - 1][:, 3:4, :, :])
        rgb = layers[n - 1][:, :3] * layers[n - 1][:, 3:4, :, :]
        for i in reversed(range(n-1)):
            rgb = rgb + layers[i][:, :3] * layers[i][:, 3:4, :, :] * alpha
            alpha = (1-layers[i][:, 3:4, :, :]) * alpha
        rgb = rgb + alpha
        return rgb

    def soft_composite(self, **kwargs):
        layers = kwargs['layers']
        z_layers = kwargs['z_layers']
        n = len(layers)

        inv_mask = (1 - layers[0][:, 3:4, :, :])
        for i in range(1, n):
            inv_mask = inv_mask * (1 - layers[i][:, 3:4, :, :])

        sum_alpha = layers[0][:, 3:4, :, :] * z_layers[0]
        for i in range(1, n):
            sum_alpha = sum_alpha + layers[i][:, 3:4, :, :] * z_layers[i]
        sum_alpha = sum_alpha + inv_mask

        inv_mask = inv_mask / sum_alpha

        rgb = layers[0][:, :3] * layers[0][:, 3:4, :, :] * z_layers[0] / sum_alpha
        for i in range(1, n):
            rgb = rgb + layers[i][:, :3] * layers[i][:, 3:4, :, :] * z_layers[i] / sum_alpha
        rgb = rgb * (1 - inv_mask) + inv_mask
        return rgb


    def soft_composite_W_bg(self, **kwargs):
        layers = kwargs['layers']
        z_layers = kwargs['z_layers']
        n = len(layers)

        sum_alpha = layers[0][:, 3:4, :, :] * z_layers[0]
        for i in range(1, n):
            sum_alpha = sum_alpha + layers[i][:, 3:4, :, :] * z_layers[i]
        sum_alpha = sum_alpha + 600

        inv_mask = 600 / sum_alpha

        rgb = layers[0][:, :3] * layers[0][:, 3:4, :, :] * z_layers[0] / sum_alpha
        for i in range(1, n):
            rgb = rgb + layers[i][:, :3] * layers[i][:, 3:4, :, :] * z_layers[i] / sum_alpha
        rgb = rgb + inv_mask
        return rgb


    def decode_and_composite(self, z: Tensor, return_overlap_loss=False, **kwargs):
        bs = z.shape[0]
        layers = []
        n = len(self.colors)
        loss = 0
        z_rnn_input = z[None, :, :].repeat(n, 1, 1)  # [len, batch size, emb dim]
        outputs, hidden = self.rnn(z_rnn_input)
        outputs = outputs.permute(1, 0, 2)  # [batch size, len, emb dim]
        outputs = outputs[:, :, :self.latent_dim] + outputs[:, :, self.latent_dim:]
        z_layers = []
        for i in range(n):
            shape_output = self.divide_shape(outputs[:, i, :])
            shape_latent = self.final_shape_latent(shape_output)
            all_points = self.decode(shape_latent)#, point_predictor=self.point_predictor[i])
            layer = self.raster(all_points, self.colors[i], verbose=kwargs['verbose'], white_background=False)
            z_pred = self.z_order(shape_output)
            layers.append(layer)
            z_layers.append(torch.exp(z_pred[:, :, None, None]))
            if return_overlap_loss:
                loss = loss + self.control_polygon_distance(all_points)

        output = self.composite_fn(layers = layers, z_layers=z_layers)
        if return_overlap_loss:
        #     overlap_alpha = layers[1][:, 3:4, :, :] + layers[2][:, 3:4, :, :]
        #     loss = F.relu(overlap_alpha - 1).mean()
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
        return output  # [:, :3]

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

    def interpolate_mini(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        z = self.interpolate_vectors(mu[0], mu[1], 10)
        output = self.decode_and_composite(z, verbose=kwargs['verbose'])
        return output

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
        n = len(self.colors)
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
                layer = self.raster(all_points_interpolate, self.colors[i], verbose=kwargs['verbose'])
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

            output = self.composite_fn(layers = layers)
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
        for i in range(7, 25):
            self.redo_features(i)
            output = self.decode_and_composite(mu, verbose=kwargs['verbose'])
            all_interpolations.append(output)
        return all_interpolations
