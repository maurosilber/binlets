from dataclasses import dataclass

import numpy as np
import torch
from models.dncnn import DnCNN
from models.unet import UnetN2N


def numpy_wrapper(model, device):
    def as_numpy(image):
        image = image.astype(np.float32)
        image = torch.tensor(image).to(device)
        image = image[None, None]
        denoised = model(image)
        return denoised[0, 0].cpu().detach().numpy()

    return as_numpy


def get_unetn2n_model(pth_path: str, device):
    model = UnetN2N(1, 1)
    model.load_state_dict(torch.load(pth_path, map_location="cpu"))
    model = model.to(device)
    model.eval()
    return numpy_wrapper(model, device)


def get_dncnn_model(pth_path: str, device):
    model = DnCNN(
        depth=17,
        n_channels=64,
        image_channels=1,
        use_bnorm=True,
        kernel_size=3,
    )
    model.load_state_dict(torch.load(pth_path, map_location="cpu"))
    model = model.to(device)
    model.eval()
    return numpy_wrapper(model, device)


@dataclass(frozen=True)
class Rescale:
    offset: float = 0.5
    scale: float = 255

    def forward(self, data):
        return data / self.scale + self.offset

    def inverse(self, data):
        return (data - self.offset) * self.scale
