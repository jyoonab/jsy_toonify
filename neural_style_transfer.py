import imutils
import cv2
import numpy as np
import torch

from model import Generator
from torchvision import transforms

def get_model_from_path(style_model_path):
    #model = cv2.dnn.readNetFromTorch(style_model_path)
    model_file = torch.load(style_model_path)
    return model_file

def style_transfer(image, model):

    device = 'cpu'
    torch.set_grad_enabled(False)
    image_size = 300 # Can be tuned, works best when the face width is between 200~250 px

    ckpt = Generator().eval().to(device)
    ckpt.load_state_dict(model)

    results = []

    image = load_image(f"images/iu.jpg", image_size)
    output = ckpt(image.to(device))
    #cv2.imwrite('./result/face_results.jpg', cv2.cvtColor(255*tensor2image(output), cv2.COLOR_BGR2RGB))

    output_array = output[0].permute(1, 2, 0).detach().cpu().numpy()

    return (0.5 * output_array + 0.5).clip(0, 1)

def load_image(path, size=None):
    image = image2tensor(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))

    w, h = image.shape[-2:]
    if w != h:
        crop_size = min(w, h)
        left = (w - crop_size)//2
        right = left + crop_size
        top = (h - crop_size)//2
        bottom = top + crop_size
        image = image[:,:,left:right, top:bottom]

    if size is not None and image.shape[-1] != size:
        image = torch.nn.functional.interpolate(image, (size, size), mode="bilinear", align_corners=True)

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
