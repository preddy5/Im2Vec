import random

import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F

from utils import fig2data, make_tensor
from .types_ import *
import pydiffvg
import math
import numpy as np
import kornia
import torchvision
import matplotlib.pyplot as plt
dsample = kornia.transform.PyrDown()

# import os
# import psutil
# process = psutil.Process(os.getpid())

class VectorVAE(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 loss_fn: str = 'MSE',
                 imsize: int = 128,
                 paths: int = 4,
                 **kwargs) -> None:
        super(VectorVAE, self).__init__()

        self.latent_dim = latent_dim
        self.imsize = imsize
        self.beta = kwargs['beta']
        self.paths = paths
        self.in_channels = in_channels
        self.scale_factor = kwargs['scale_factor']
        self.learn_sampling = kwargs['learn_sampling']
        self.only_auxillary_training = kwargs['only_auxillary_training']
        self.memory_leak_training = kwargs['memory_leak_training']

        if loss_fn == 'BCE':
            self.loss_fn = F.binary_cross_entropy_with_logits
        else:
            self.loss_fn = F.mse_loss
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    # nn.BatchNorm2d(h_dim),
                    nn.ReLU())
            )
            in_channels = h_dim
        outsize = int(imsize/(2**5))
        self.fc_mu = nn.Linear(hidden_dims[-1]*outsize*outsize, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*outsize*outsize, latent_dim)
        self.encoder = nn.Sequential(*modules)

        self.circle_rad = kwargs['radius']
        self.number_of_points = self.paths * 3

        sample_rate = 1
        angles = torch.arange(0, self.number_of_points, dtype=torch.float32) *6.28319/ self.number_of_points
        id = self.sample_circle(self.circle_rad, angles, sample_rate)
        base_control_features = torch.tensor([[1,0],[0,1],[0,1]], dtype=torch.float32)
        self.id = id[:,:]
        self.angles = angles
        self.register_buffer('base_control_features', base_control_features)
        self.deformation_range = 6.28319/ 4

        def get_computational_unit(in_channels, out_channels, unit):
            if unit=='conv':
                return nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2, padding_mode='circular', stride=1, dilation=1)
            else:
                return nn.Linear(in_channels, out_channels)
            # Build Decoder

        unit='conv'
        if unit=='conv':
            self.decode_transform = lambda x: x.permute(0, 2, 1)
        else:
            self.decode_transform = lambda x: x
        num_one_hot = base_control_features.shape[1]
        fused_latent_dim = latent_dim + num_one_hot+ (sample_rate*2)
        self.decoder_input = get_computational_unit(fused_latent_dim, fused_latent_dim*2, unit)
        # self.point_predictor = nn.ModuleList([
        #     get_computational_unit(latent_dim*2 + num_one_hot, latent_dim*2, unit),
        #     get_computational_unit(latent_dim*3 + num_one_hot, latent_dim*2, unit),
        #     get_computational_unit(latent_dim*3 + num_one_hot, latent_dim*2, unit),
        #     get_computational_unit(latent_dim*3 + num_one_hot, 2, unit),
        #     # nn.Sigmoid()  # bound spatial extent
        # ])
        self.point_predictor = nn.ModuleList([
            get_computational_unit(fused_latent_dim*2, fused_latent_dim*2, unit),
            get_computational_unit(fused_latent_dim*2, fused_latent_dim*2, unit),
            get_computational_unit(fused_latent_dim*2, fused_latent_dim*2, unit),
            get_computational_unit(fused_latent_dim*2, fused_latent_dim*2, unit),
            get_computational_unit(fused_latent_dim*2, 2, unit),
            # nn.Sigmoid()  # bound spatial extent
        ])
        self.render = pydiffvg.RenderFunction.apply
        if self.learn_sampling:
            self.sample_deformation = nn.Sequential(
                get_computational_unit(latent_dim + 2+ (sample_rate*2), latent_dim*2, unit),
                nn.ReLU(),
                get_computational_unit(latent_dim * 2, latent_dim * 2, unit),
                nn.ReLU(),
                get_computational_unit(latent_dim*2, 1, unit),
            )
        self.aux_network = nn.Sequential(
            get_computational_unit(latent_dim, latent_dim*2, 'mlp'),
            nn.LeakyReLU(),
            get_computational_unit(latent_dim * 2, latent_dim * 2, 'mlp'),
            nn.LeakyReLU(),
            get_computational_unit(latent_dim * 2, latent_dim * 2, 'mlp'),
            nn.LeakyReLU(),
            get_computational_unit(latent_dim*2, 3, 'mlp'),
        )
        self.latent_lossvpath = {}
        self.save_lossvspath = False
        if self.only_auxillary_training:
            self.save_lossvspath = True
            for name, param in self.named_parameters():
                if 'aux_network' in name:
                    print(name)
                    param.requires_grad =True
                else:
                    param.requires_grad =False
        # self.lpips = VGGPerceptualLoss(False)

    def redo_features(self, n):
        self.paths = n
        self.number_of_points = self.paths * 3
        self.angles = (torch.arange(0, self.number_of_points, dtype=torch.float32) *6.28319/ self.number_of_points)

        id = self.sample_circle(self.circle_rad, self.angles, 1)
        self.id = id[:,:]

    def sample_circle(self, r, angles, sample_rate=10):
        pos = []
        for i in range(1, sample_rate+1):
            x = (torch.cos(angles*(sample_rate/i)) * r)# + r
            y = (torch.sin(angles*(sample_rate/i)) * r)# + r
            pos.append(x)
            pos.append(y)
        return torch.stack(pos, dim=-1)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def raster(self, all_points, color=[0,0,0, 1], verbose=False, white_background=True):
        assert len(color) == 4
        # print('1:', process.memory_info().rss*1e-6)
        render_size = self.imsize
        bs = all_points.shape[0]
        if verbose:
            render_size = render_size*2
        outputs = []
        all_points = all_points*render_size
        num_ctrl_pts = torch.zeros(self.paths, dtype=torch.int32) + 2
        color = make_tensor(color)
        for k in range(bs):
            # Get point parameters from network
            shapes = []
            shape_groups = []
            points = all_points[k].cpu().contiguous()#[self.sort_idx[k]]

            if verbose:
                np.random.seed(0)
                colors = np.random.rand(self.paths, 4)
                high = np.array((0.565, 0.392, 0.173, 1))
                low = np.array((0.094, 0.310, 0.635, 1))
                diff = (high-low)/(self.paths)
                colors[:, 3] = 1
                for i in range(self.paths):
                    scale = diff*i
                    color = low + scale
                    color[3] = 1
                    color = torch.tensor(color)
                    num_ctrl_pts = torch.zeros(1, dtype=torch.int32) + 2
                    if i*3 + 4 > self.paths * 3:
                        curve_points = torch.stack([points[i*3], points[i*3+1], points[i*3+2], points[0]])
                    else:
                        curve_points = points[i*3:i*3 + 4]
                    path = pydiffvg.Path(
                        num_control_points=num_ctrl_pts, points=curve_points,
                        is_closed=False, stroke_width=torch.tensor(4))
                    path_group = pydiffvg.ShapeGroup(
                        shape_ids=torch.tensor([i]),
                        fill_color=None,
                        stroke_color=color)
                    shapes.append(path)
                    shape_groups.append(path_group)
                for i in range(self.paths * 3):
                    scale = diff*(i//3)
                    color = low + scale
                    color[3] = 1
                    color = torch.tensor(color)
                    if i%3==0:
                        # color = torch.tensor(colors[i//3]) #green
                        shape = pydiffvg.Rect(p_min = points[i]-8,
                                             p_max = points[i]+8)
                        group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([self.paths+i]),
                                                           fill_color=color)

                    else:
                        # color = torch.tensor(colors[i//3]) #purple
                        shape = pydiffvg.Circle(radius=torch.tensor(8.0),
                                                 center=points[i])
                        group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([self.paths+i]),
                                                           fill_color=color)
                    shapes.append(shape)
                    shape_groups.append(group)

            else:

                path = pydiffvg.Path(
                    num_control_points=num_ctrl_pts, points=points,
                    is_closed=True)

                shapes.append(path)
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(shapes) - 1]),
                    fill_color=color,
                    stroke_color=color)
                shape_groups.append(path_group)
            scene_args = pydiffvg.RenderFunction.serialize_scene(render_size, render_size, shapes, shape_groups)
            out = self.render(render_size,  # width
                         render_size,  # height
                         2,  # num_samples_x
                         2,  # num_samples_y
                         102,  # seed
                         None,
                         *scene_args)
            out = out.permute(2, 0, 1).view(4, render_size, render_size)#[:3]#.mean(0, keepdim=True)
            outputs.append(out)
        output =  torch.stack(outputs).to(all_points.device)
        alpha = output[:, 3:4, :, :]

        # map to [-1, 1]
        if white_background:
            output_white_bg = output[:, :3, :, :]*alpha + (1-alpha)
            output = torch.cat([output_white_bg, alpha], dim=1)
        del num_ctrl_pts, color
        return output

    def decode(self, z: Tensor, verbose=False) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        self.id = self.id.to(z.device)

        bs = z.shape[0]
        z = z[:, None, :].repeat([1, self.paths *3, 1])
        base_control_features = self.base_control_features[None, :, :].repeat(bs, self.paths, 1 )
        z_base = torch.cat([z, base_control_features], dim=-1)
        z_base_transform = self.decode_transform(z_base)
        if self.learn_sampling:
            self.angles = self.angles.to(z.device)
            angles= self.angles[None, :, None].repeat(bs, 1, 1)
            x = torch.cos(angles)# + r
            y = torch.sin(angles)# + r
            z_angles = torch.cat([z_base, x, y], dim=-1)

            angles_delta = self.sample_deformation(self.decode_transform(z_angles))
            angles_delta = F.tanh(angles_delta/50)*self.deformation_range
            angles_delta = self.decode_transform(angles_delta)

            new_angles = angles + angles_delta
            x = (torch.cos(new_angles) * self.circle_rad)# + r
            y = (torch.sin(new_angles) * self.circle_rad)# + r
            z = torch.cat([z_base, x, y], dim=-1)
        else:
            id = self.id[None, :, :].repeat(bs, 1, 1)
            z = torch.cat([z_base, id], dim=-1)

        all_points = self.decoder_input(self.decode_transform(z))
        for compute_block in self.point_predictor:
            all_points = F.relu(all_points)
            # all_points = torch.cat([z_base_transform, all_points], dim=1)
            all_points = compute_block(all_points)
        all_points = self.decode_transform(F.sigmoid(all_points/self.scale_factor))
        return all_points

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu#eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        all_points = self.decode(z)
        output = self.raster(all_points)
        return  [output, input, mu, log_var]

    def bilinear_downsample(self, tensor, size):
        return torch.nn.functional.interpolate(tensor, size, mode='bilinear')

    def gaussian_pyramid_loss(self, recons, input):
        recon_loss =self.loss_fn(recons, input, reduction='none').mean(dim=[1,2,3]) #+ self.lpips(recons, input)*0.1
        for j in range(2,5):
            recons = dsample(recons)
            input = dsample(input)
            recon_loss = recon_loss + self.loss_fn(recons, input, reduction='none').mean(dim=[1,2,3])/j
        return recon_loss

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0][:, :3, :, :]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        other_losses = 0
        if len(args)==5:
            other_losses = args[4]
        aux_loss = 0
        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recon_loss = self.gaussian_pyramid_loss(recons, input)
        if self.only_auxillary_training:
            recon_loss_non_reduced = recon_loss[:, None].clone().detach()
            spacing = self.aux_network(mu.clone().detach())
            latents = mu.cpu().numpy()
            num_latents = latents.shape[0]
            if self.save_lossvspath:
                recon_loss_non_reduced_cpu = recon_loss_non_reduced.cpu().numpy()
                keys  = self.latent_lossvpath.keys()
                for i in range(num_latents):
                    if np.array2string(latents[i]) in keys:
                        pair = make_tensor([self.paths, recon_loss_non_reduced_cpu[i, 0], ])[None, :].to(mu.device)
                        self.latent_lossvpath[np.array2string(latents[i])]\
                            = torch.cat([self.latent_lossvpath[np.array2string(latents[i])], pair], dim=0)
                    else:
                        self.latent_lossvpath[np.array2string(latents[i])] = make_tensor([[self.paths, recon_loss_non_reduced_cpu[i, 0]], ]).to(mu.device)
                num = torch.ones_like(spacing[:, 0]) * self.paths
                est_loss = spacing[:,2] + 1/torch.exp(num*spacing[:,0] - spacing[:,1])
                # est_loss = spacing[:, 2] + (spacing[i, 0] / num)

                aux_loss = torch.abs((est_loss - recon_loss_non_reduced)).mean() * 10
            else:
                aux_loss = 0
                for i in range(num_latents):
                    pair = self.latent_lossvpath[np.array2string(latents[i])]
                    est_loss = spacing[i, 2] + 1 / torch.exp(pair[:, 0] * spacing[i, 0] - spacing[i, 1])

                    # est_loss = spacing[i, 2] + (spacing[i, 0] / pair[:, 0])
                    aux_loss = aux_loss + torch.abs((est_loss - pair[:, 1])).mean()
            loss =  aux_loss
            kld_loss = 0#self.beta*kld_weight * kld_loss
            logs = {'Reconstruction_Loss': recon_loss.mean(), 'KLD': -kld_loss, 'aux_loss': aux_loss}
            return {'loss': loss, 'progress_bar': logs }
        recon_loss = recon_loss.mean()
        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = 0#self.beta*kld_weight * kld_loss
        recon_loss = recon_loss*10
        loss =  recon_loss + kld_loss + other_losses
        logs = {'Reconstruction_Loss': recon_loss, 'KLD': -kld_loss, 'aux_loss': aux_loss, 'other losses': other_losses}
        return {'loss': loss, 'progress_bar': logs }

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        all_points = self.decode(z)
        samples = self.raster(all_points)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return  self.raster(self.decode(z), verbose=random.choice([True, False]))
 # .type(torch.FloatTensor).to(device)

    def save(self, x, save_dir, name):
        z, log_var = self.encode(x)
        all_points = self.decode(z)
        # print(all_points.std(dim=1))
        # all_points = ((all_points-0.5)*2 + 0.5)*self.imsize
        # if type(self.sort_idx) == type(None):
        #     angles = torch.atan(all_points[:,:,1]/all_points[:,:,0]).detach()
        #     self.sort_idx = torch.argsort(angles, dim=1)
        # Process the batch sequentially
        outputs = []
        for k in range(1):
            # Get point parameters from network
            shapes = []
            shape_groups = []
            points = all_points[k].cpu()#[self.sort_idx[k]]

            color = torch.cat([torch.tensor([0,0,0,1]),])
            num_ctrl_pts = torch.zeros(self.paths, dtype=torch.int32) + 2

            path = pydiffvg.Path(
                num_control_points=num_ctrl_pts, points=points,
                is_closed=True)

            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=color,
                stroke_color=color)
            shape_groups.append(path_group)
            pydiffvg.save_svg(f"{save_dir}{name}/{name}.svg",
                              self.imsize, self.imsize, shapes, shape_groups)

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
            all_points = self.decode(z)
            all_interpolations.append(self.raster(all_points, verbose=kwargs['verbose']))
        return all_interpolations

    def interpolate2D(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        all_interpolations = []
        y_axis = self.interpolate_vectors(mu[7], mu[6], 10)
        for i in range(10):
            z = self.interpolate_vectors(y_axis[i], mu[3], 10)
            all_points = self.decode(z)
            all_interpolations.append(self.raster(all_points, verbose=kwargs['verbose']))
        return all_interpolations


    def naive_vector_interpolate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        all_points = self.decode(mu)
        all_interpolations = []
        for i in range(mu.shape[0]):
            z = self.interpolate_vectors(all_points[2], all_points[i], 10)
            all_interpolations.append(self.raster(z, verbose=kwargs['verbose']))
        return all_interpolations

    def visualize_sampling(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        all_interpolations = []
        for i in range(5,27):
            self.redo_features(i)
            all_points = self.decode(mu)
            all_interpolations.append(self.raster(all_points, verbose=kwargs['verbose']))
        return all_interpolations

    def sampling_error(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        error = []
        figure = plt.figure(figsize=(6, 6))
        bs = x.shape[0]
        for i in range(7,25):
            self.redo_features(i)
            results = self.forward(x)
            recons = results[0][:, :3, :, :]
            input_batch = results[1]

            recon_loss = self.gaussian_pyramid_loss(recons, input_batch)
            print(recon_loss)
            error.append(recon_loss)
        etn = torch.stack(error, dim=1).numpy()
        np.savetxt('sample_error.csv', etn, delimiter=',')
        y = np.arange(7,25)
        for i in range(bs):
            plt.plot(y, etn[i,:], label=str(i+1))
        plt.legend(loc='upper right')
        img = fig2data(figure)
        return img

    def visualize_aux_error(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        bs = mu.shape[0]
        all_spacing = []
        figure = plt.figure(figsize=(6, 6))

        for i in np.arange(7,25):
            spacing = self.aux_network(mu.clone().detach())
            num = torch.ones_like(spacing[:,0])*i
            # est_loss = spacing[:,2] + 1/torch.exp(num*spacing[:,0] + spacing[:,1])
            est_loss =     spacing[:,2] + (spacing[:,0]/num)

            # print(i, spacing[0])
            all_spacing.append(est_loss)
        all_spacing = torch.stack(all_spacing, dim=1).detach().cpu().numpy()
        y = np.arange(7,25)
        for i in range(bs):
            plt.plot(y, all_spacing[i,:], label=str(i+1))
        plt.legend(loc='upper right')
        img = fig2data(figure)
        return img
