import threading
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import imutils
from neural_style_transfer import get_model_from_path, style_transfer, image2tensor, tensor2image, imshow
from data import *
from torchvision import transforms
import tempfile
import av

from CutGan.CUTGan import CUTGan
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings, VideoProcessorBase, WebRtcMode

'''
from streamlit_webrtc import (
    AudioProcessorBase,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)'''

cut_gan = CUTGan('./CutGAN/images\\4.png')

def assignment_one_page(gan_type_name, convert_target, style_model_name):
    if gan_type_name == 'AnimeGAN':
        if convert_target == 'Image':
            image_input(gan_type_name, style_model_name)
        elif convert_target == 'Webcam':
            webcam_input(gan_type_name, style_model_name)

    if gan_type_name == 'CutGAN':
        if convert_target == 'Image':
            image_input(gan_type_name, style_model_name)
        elif convert_target == 'Webcam':
            webcam_input(gan_type_name, style_model_name)

def image_input(gan_type_name, style_model_name):
    '''Upload or Image_Select'''
    if st.sidebar.checkbox('Upload'):
        content_file = st.sidebar.file_uploader("Choose a Content Image", type=["png", "jpg", "jpeg"])
    else:
        content_name = st.sidebar.selectbox("Choose the content images:", content_images_name)
        content_file = content_images_dict[content_name]

    if content_file is not None:
        content = Image.open(content_file)
        content = np.array(content) #pil to cv
        content = cv2.cvtColor(content, cv2.COLOR_RGB2BGR)
    else:
        st.warning("Upload an Image OR Untick the Upload Button")
        st.stop()

    orig_h, orig_w = content.shape[0:2]

    '''Image Quality Side Bar'''
    # Sidebar Decides the size of the image (bigger images means better quality)
    WIDTH = st.sidebar.select_slider('QUALITY (May reduce the speed)', list(range(150, 501, 50)), value=150)
    content = imutils.resize(content, width=WIDTH)

    '''Image Convert Start Point'''
    if gan_type_name == 'AnimeGAN':
        '''Initialize the model'''
        style_model_path = animegan_model_dict[style_model_name]
        model = get_model_from_path(style_model_path)

        '''Start Converting Image'''
        generated = style_transfer(content, model)
    elif gan_type_name == 'CutGAN':
        '''Initialize the model'''
        style_model_path = cutgan_model_dict[style_model_name]

        '''Start Converting Image'''
        content = cv2.resize(content, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        generated = cut_gan.start_converting(content)
        generated = cv2.resize(generated, dsize=(orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    st.sidebar.image(content, width=300, channels='BGR')
    st.image(generated, channels='BGR', clamp=True, caption="Result Image")


def webcam_input(gan_type_name, style_model_name):
    st.header("Webcam Live Feed")
    WIDTH = st.sidebar.select_slider('QUALITY (May reduce the speed)', list(range(150, 501, 50)))

    class NeuralStyleTransferTransformer(VideoProcessorBase):
        def __init__(self) -> None:
            self._width = WIDTH

            if gan_type_name == 'AnimeGAN':
                self.style_model_path = animegan_model_dict[style_model_name]
                self.model = get_model_from_path(self.style_model_path)

            #elif gan_type_name == 'CutGAN':
            #    self.style_model_path = cutgan_model_dict[style_model_name]

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            orig_h, orig_w = img.shape[0:2]

            if gan_type_name == 'AnimeGAN':
                generated = style_transfer(img, self.model)
                generated = cv2.convertScaleAbs(generated, alpha=255.0)

            elif gan_type_name == 'CutGAN':
                img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
                generated = cut_gan.start_converting(img)
                generated = cv2.resize(generated, dsize=(orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                generated = cv2.cvtColor(generated, cv2.COLOR_RGB2BGR)

            return av.VideoFrame.from_ndarray(generated)

        def set_width(self, width):
            update_needed = self._width != width
            self._width = width
            if update_needed:
                self._update_model()

        def update_model_name(self, model_name):
            update_needed = self._model_name != model_name
            self._model_name = model_name
            if update_needed:
                self._update_model()

        def _update_model(self):
            print("state changed ", gan_type_name)
            self.style_model_path = animegan_model_dict[style_model_name]
            with self._model_lock:
                self.model = get_model_from_path(self.style_model_path)

    ctx = webrtc_streamer(
        rtc_configuration=ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        ),
        media_stream_constraints={"video": True, "audio": False},
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=NeuralStyleTransferTransformer,
        key="neural-style-transfer",
    )
    if ctx.video_transformer:
        ctx.video_transformer.set_width(WIDTH)
