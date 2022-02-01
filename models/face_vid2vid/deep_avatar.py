import cv2
import numpy as np
import torch
from moviepy.editor import *
from skimage import img_as_ubyte
from skimage.transform import resize

from models.face_vid2vid.utils import get_checkpoints, \
    image_translation, video_translation, frame_translation, find_best_frame


def get_image_array_from_path(image_path: str) -> np.ndarray:
    avatar_image = cv2.imread(image_path)
    avatar_image = cv2.cvtColor(avatar_image, cv2.COLOR_BGR2RGB)
    return resize(avatar_image, (256, 256))[..., :3]

def get_frame_list_from_path(video_path: str) -> (np.ndarray, int):
    video_reader = imageio.get_reader(video_path)
    fps = video_reader.get_meta_data()['fps']
    frame_list = [frame for frame in video_reader]
    video_reader.close()
    return [resize(frame, (256, 256))[..., :3] for frame in frame_list], fps


class DeepAvatar:
    def __init__(self) -> None:
        super(DeepAvatar, self).__init__()
        self.model_name: str = 'DeepAvatar'
        self.model_version: str = '1.0.2'
        self.model_root_path: str = './models/face_vid2vid'

        # load pretrained model
        self.generator, self.kp_detector, self.he_estimator = self.load_pretrained_model()

    def load_pretrained_model(self) -> (object, object, object):
        return get_checkpoints(config_path=f'{self.model_root_path}/config/vox-256-spade.yaml',
                               checkpoint_path=f'{self.model_root_path}/ckpt/00000189-checkpoint.pth.tar')

    def change_head_pose(self, image_path: str, yaw: int = 0, pitch: int = 0, roll: int = 0):
        image_array = get_image_array_from_path(image_path=image_path)
        options_dict: dict = {
            'avatar_image': image_array,
            'reference_image': image_array,
            'generator': self.generator,
            'kp_detector': self.kp_detector,
            'he_estimator': self.he_estimator,
            'relative': True,
            'adapt_movement_scale': True,
            'estimate_jacobian': False,
            'cpu_mode': False,
            'free_view': True,
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll
        }
        prediction_result = image_translation(options_dict)
        return img_as_ubyte(prediction_result)

    def get_action_list_from_video(self, image_path: str, action: str = "talk",
                                   yaw: int = 0, pitch: int = 0, roll: int = 0, hq_mode: bool = False) -> (list, int):
        image_array = get_image_array_from_path(image_path=image_path)
        frame_list, fps = get_frame_list_from_path(video_path=f'./asset/action/{action}.mp4')

        if hq_mode:
            best_frame_index = find_best_frame(avatar_image=image_array,
                                               frame_list=frame_list,
                                               cpu_mode=False)
            action_forward = resized_frame_list[best_frame_index:]
            action_backward = resized_frame_list[:(best_frame_index + 1)][::-1]

            # make options
            options_dict: dict = {
                'avatar_image': image_array,
                'action_video': action_forward,
                'generator': self.generator,
                'kp_detector': self.kp_detector,
                'he_estimator': self.he_estimator,
                'relative': True,
                'adapt_movement_scale': True,
                'estimate_jacobian': False,
                'cpu_mode': False,
                'free_view': False,
                'yaw': yaw,
                'pitch': pitch,
                'roll': roll
            }

            # prediction
            predictions_forward = video_translation(options_dict)
            options_dict['action_video'] = action_backward
            predictions_backward = video_translation(options_dict)
            prediction_result = predictions_backward[::-1] + predictions_forward[1:]

        else:
            # make options
            options_dict: dict = {
                'avatar_image': image_array,
                'action_video': frame_list,
                'generator': self.generator,
                'kp_detector': self.kp_detector,
                'he_estimator': self.he_estimator,
                'relative': True,
                'adapt_movement_scale': True,
                'estimate_jacobian': False,
                'cpu_mode': False,
                'free_view': False,
                'yaw': yaw,
                'pitch': pitch,
                'roll': roll
            }
            prediction_result = video_translation(options_dict)

        return [img_as_ubyte(frame) for frame in prediction_result], int(fps)

    def get_action_frame_from_webcam(self, synchronizer, avatar_tensor, frame):
        options_dict: dict = {
            'avatar_tensor': avatar_tensor,
            'frame': frame,
            'generator': self.generator,
            'kp_detector': self.kp_detector,
            'he_estimator': self.he_estimator,
            'relative': True,
            'adapt_movement_scale': True,
            'estimate_jacobian': False,
            'cpu_mode': False,
            'free_view': False,
            'yaw': 0,
            'pitch': 0,
            'roll': 0,
            'kp_canonical': synchronizer.kp_canonical,
            'kp_source': synchronizer.kp_source,
            'kp_driving_initial': synchronizer.kp_driving_initial
        }

        return frame_translation(options_dict)
