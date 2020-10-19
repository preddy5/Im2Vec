import glob
import os

import torch
import math
import svgpathtools
import pydiffvg
import argparse
import numpy as np
import torchvision.utils as vutils
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([  # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation([0, 360], resample=3, fill=(255,255,255)),
    # transforms.RandomAffine([0, 0], (0.0,0.05), (1.0,1.0), resample=3, fillcolor=(255,255,255)),
    transforms.ToTensor(),
])

def load_img(filename):
    x = Image.open(filename).convert('RGB')
    return transform(x)[None, :, :, :]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg", help="source SVG path")
    args = parser.parse_args()
    svg_folder = os.path.join(args.svg)
    svgs = glob.glob(svg_folder+'/*.png')
    renders = []
    for file in range(len(svgs)):
        name = svg_folder+f'/{file}.png'
        print(name)
        tensor = load_img(name)
        renders.append(tensor)
    render = torch.cat(renders, dim=0)
    vutils.save_image(render.cpu().data,
                      svg_folder+
                      f"/img.png",
                      normalize=False,
                      nrow=10)