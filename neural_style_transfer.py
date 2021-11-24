import imutils
import cv2
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from model import Generator
from torchvision import transforms
from PIL import Image

def get_model_from_path(style_model_path):
    model_file = torch.load(style_model_path)
    return model_file

def style_transfer(image, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(False)

    ckpt = Generator().eval().to(device)
    ckpt.load_state_dict(model)

    image = image2tensor(image)

    output = ckpt(image.to(device))

    return tensor2image(output)

def load_image(content_image, size=None):
    image = AnimeGAN_v2.image2tensor(content_image)
    return image

def image2tensor(image):
    image = torch.FloatTensor(image).permute(2,0,1).unsqueeze(0)/255.
    return (image-0.5)/0.5

def tensor2image(tensor):
    tensor = tensor.clamp_(-1., 1.).detach().squeeze().permute(1,2,0).cpu().numpy()
    return tensor*0.5 + 0.5

def imshow(img, size=5, cmap='jet'):
    plt.figure(figsize=(size,size))
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()
