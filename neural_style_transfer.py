import imutils
import cv2
import numpy as np
import torch
import os

from model import Generator
from torchvision import transforms
from PIL import Image

def get_model_from_path(style_model_path):
    #model = cv2.dnn.readNetFromTorch(style_model_path)
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

def generate_new_video(video, model):
    output_path: str = './output'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path = f"{output_path}/anime_output.mp4"

    # load video
    video_path = video
    video_output_size = (450, 250)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    output_video = cv2.VideoWriter(f'{output_path}', fourcc, cap.get(cv2.CAP_PROP_FPS), video_output_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, video_output_size)
            # output_frame = cv2.flip(frame, 0)
            output_frame = generate_new_frame(frame, model)
            output_frame = cv2.convertScaleAbs(output_frame, alpha=255.0)

            output_video.write(output_frame)
            cv2.imshow("frame", frame)
            cv2.imshow("output_frame", output_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

    return output_path

def generate_new_frame(frame, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(False)

    ckpt = Generator().eval().to(device)
    ckpt.load_state_dict(model)

    content_image = image2tensor(frame)
    output = ckpt(content_image.to(device))
    output_array = output[0].permute(1, 2, 0).detach().cpu().numpy()
    return (0.5 * output_array + 0.5).clip(0, 1)

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
