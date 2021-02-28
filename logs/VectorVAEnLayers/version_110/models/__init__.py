from .base import *
from .vanilla_vae import *
from .vector_vae import VectorVAE
from .vector_vae_nlayers import VectorVAEnLayers


# Aliases
VAE = VanillaVAE


vae_models = {'VanillaVAE':VanillaVAE,
              'VectorVAE':VectorVAE,
              'VectorVAEnLayers': VectorVAEnLayers}
