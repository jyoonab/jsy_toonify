import cv2
import torch
import streamlit as st
import numpy as np
from skimage.transform import resize
from models.face_vid2vid.deep_avatar import get_image_array_from_path
from models.anime_gan.neural_style_transfer import style_transfer
import mediapipe as mp

from datetime import datetime

class Vid2vid:
    def __init__(self) -> None:
        print("Initializing Vid2vid")
        self.deep_avatar = self.get_deep_avatar_model()
        self.avatar_synchronizer = self.get_avatar_synchronizer()

        self.image_array = get_image_array_from_path(image_path="./models/face_vid2vid/asset/avatar/3.png")
        self.avatar_tensor = torch.tensor(self.image_array[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).cuda()

    def get_avatar_synchronizer(self):
        from models.face_vid2vid.avatar_synchronizer import AvatarSynchronizer
        return AvatarSynchronizer()

    @st.cache
    def get_deep_avatar_model(self):
        from models.face_vid2vid.deep_avatar import DeepAvatar
        return DeepAvatar()

    def start_converting(self, input_image):
        if not self.avatar_synchronizer.activated:
            self.avatar_synchronizer.activate(self.avatar_tensor, input_image, self.deep_avatar.kp_detector, self.deep_avatar.he_estimator)
        return self.deep_avatar.get_action_frame_from_webcam(synchronizer=self.avatar_synchronizer, avatar_tensor=self.avatar_tensor, frame=input_image)

    def change_target_image_from_array(self, input_image):
        resized_input_image = resize(input_image, (256, 256))
        self.avatar_tensor = torch.tensor(resized_input_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).cuda()

    def change_target_image_from_image(self, image_path):
        image_array = get_image_array_from_path(image_path=image_path)
        self.avatar_tensor = torch.tensor(image_array[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).cuda()

class CUT:
    def __init__(self) -> None:
        print("Initializing CUT")
        #start = datetime.now()
        self.model = self.get_model()
        #print("Time Taken: ", datetime.now() - start)

    #@st.cache
    def get_model(self):
        from models.CUT.CUT import CUT
        return CUT()

    def start_converting(self, input_image):
        return self.model.start_converting(input_image)

    def get_image_array_from_path(image_path: str) -> np.ndarray:
        avatar_image = cv2.imread(image_path)
        avatar_image = cv2.cvtColor(avatar_image, cv2.COLOR_BGR2RGB)
        return resize(avatar_image, (512, 512))[..., :3]

    def start_converting_background_only(self, input_image, filter):
        selfie_segmentator: object = mp.solutions.selfie_segmentation

        if filter == "Self":
            generated_image = self.model.start_converting(input_image)
        elif filter == "New York":
            generated_image = get_image_array_from_path("./src/new_york.png")
            generated_image = np.round(255 * generated_image).astype(np.uint8)
            generated_image = cv2.resize(generated_image, dsize=(512, 512), interpolation=cv2.INTER_LANCZOS4)
        elif filter == "Bang":
            generated_image = get_image_array_from_path("./src/bang.png")
            generated_image = np.round(255 * generated_image).astype(np.uint8)
            generated_image = cv2.resize(generated_image, dsize=(512, 512), interpolation=cv2.INTER_LANCZOS4)

        input_image = cv2.resize(input_image, dsize=(512, 512), interpolation=cv2.INTER_LANCZOS4)

        '''Fetch Body Shape by using Selfie_segmentation'''
        with selfie_segmentator.SelfieSegmentation(model_selection=1) as selfie_segmentator:
            segmented_img = selfie_segmentator.process(input_image)
            mask = segmented_img.segmentation_mask

            thr, mask_black_white = cv2.threshold(mask, 0.7, 255.0, cv2.THRESH_BINARY)
            mask_converted = cv2.normalize(src=mask_black_white, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            blurred_mask_converted = cv2.GaussianBlur(mask_converted,(15,15),0)
            background_mask = cv2.bitwise_not(mask_converted)

            human_shape = cv2.bitwise_and(input_image, input_image, mask=blurred_mask_converted)
            background = cv2.bitwise_and(generated_image, generated_image, mask=background_mask)
            result_img_np = cv2.add(background, human_shape)

            return result_img_np

        return input_image


class AnimeGan:
    def __init__(self) -> None:
        print("Initializing AnimeGan")
        self.face_paint_model, self.portrait_model, self.your_name_model = self.get_model()

    @st.cache
    def get_model(self):
        from models.anime_gan.neural_style_transfer import get_model_from_path
        face_paint_model = get_model_from_path('models/anime_gan/models\\face_paint_512_v2_0.pt')
        portrait_model = get_model_from_path('models/anime_gan/models\\portrait.pt')
        your_name_model = get_model_from_path('models/anime_gan/models\\weater_and_yourname.pt')
        return face_paint_model, portrait_model, your_name_model


    def start_converting(self, input_image, model_name):
        img = cv2.resize(input_image, dsize=(720, 720), interpolation=cv2.INTER_LANCZOS4)
        if model_name == 'Portrait':
            return style_transfer(img, self.portrait_model)
        elif model_name == 'Your Name':
            return style_transfer(img, self.your_name_model)
        return style_transfer(img, self.face_paint_model)
