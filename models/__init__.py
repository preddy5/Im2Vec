from .base import *
from .vanilla_vae import *
from .vector_vae import VectorVAE


# Aliases
VAE = VanillaVAE


vae_models = {'VanillaVAE':VanillaVAE,
              'VectorVAE':VectorVAE}
