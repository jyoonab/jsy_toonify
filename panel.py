import threading
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import imutils
from neural_style_transfer import get_model_from_path, style_transfer, image2tensor, tensor2image, imshow
from data_list import *
from torchvision import transforms
import tempfile
import av
import os
import time

from CutGan.CUTGan import CUTGan
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings, VideoProcessorBase, WebRtcMode
from GPEN.face_enhancement import FaceEnhancement

os.environ['KMP_DUPLICATE_LIB_OK']='True' # prevents issue (https://jaeniworld.tistory.com/8)
cut_gan = CUTGan('./CutGAN/images\\4.png', 'faces2comics_512_CUT')
faceenhancer = FaceEnhancement(size=256, model='GPEN-BFR-256', use_sr=True, sr_model='rrdb_realesrnet_psnr', channel_multiplier=1, narrow=0.5, device='cuda')
#faceenhancer = FaceEnhancement(size=512, model='GPEN-BFR-512', use_sr=True, sr_model='rrdb_realesrnet_psnr', channel_multiplier=2, narrow=1, device='cuda')

def project_summary_page():
    st.header("1. Cartoonizer Demo")

def cartoonizer_demo_page(gan_type_name, convert_target, style_model_name):
    if convert_target == 'Image':
        image_input(gan_type_name, style_model_name)
    elif convert_target == 'Webcam':
        webcam_input(gan_type_name, style_model_name)

def image_input(gan_type_name, style_model_name, photo_to_cartoon_time = 0, gpen_to_cartoon_time = 0, photo_to_gpen_time = 0):
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

    '''Image Quality Bar'''
    WIDTH = st.select_slider('QUALITY (May reduce the speed)', list(range(150, 501, 50)), value=300)
    content = imutils.resize(content, width=WIDTH)
    orig_h, orig_w = content.shape[0:2]
    start_time = time.time()
    content_gpen, orig_faces, enhanced_faces = faceenhancer.process(content)
    end_time = time.time()
    photo_to_gpen_time = np.round(end_time - start_time, 3)

    '''Image Convert Start Point'''
    if gan_type_name == 'AnimeGAN_v2' or gan_type_name == 'AnimeGAN':
        '''Initialize the model'''
        if gan_type_name == 'AnimeGAN_v2':
            style_model_path = animegan_v2_model_dict[style_model_name]
        elif gan_type_name == 'AnimeGAN':
            style_model_path = animegan_model_dict[style_model_name]

        model = get_model_from_path(style_model_path)

        '''Start Converting Image(no GPEN)'''
        start_time = time.time()
        generated = style_transfer(content, model)
        end_time = time.time()
        photo_to_cartoon_time = np.round(end_time - start_time, 3)

        '''Start Converting Image(with GPEN)'''
        start_time = time.time()
        generated_gpen = style_transfer(content_gpen, model)
        end_time = time.time()
        gpen_to_cartoon_time = np.round(end_time - start_time, 3)

    elif gan_type_name == 'CutGAN':
        '''Initialize the model'''
        style_model_path = cutgan_model_dict[style_model_name]

        '''Start Converting Image(no GPEN)'''
        start_time = time.time()
        content = cv2.resize(content, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        generated = cut_gan.start_converting(content)
        generated = cv2.resize(generated, dsize=(orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        end_time = time.time()
        photo_to_cartoon_time = np.round(end_time - start_time, 3)

        '''Start Converting Image(with GPEN)'''
        start_time = time.time()
        content_gpen = cv2.resize(content_gpen, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        generated_gpen = cut_gan.start_converting(content_gpen)
        generated_gpen = cv2.resize(generated_gpen, dsize=(orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        end_time = time.time()
        gpen_to_cartoon_time = np.round(end_time - start_time, 3)

    '''Display Image Test Result without GPEN'''
    image_orig_col, image_result_col = st.columns(2)
    with image_orig_col:
        st.image(content, width=300, channels='BGR', clamp=True, caption="Original Image")
    with image_result_col:
        st.image(generated, width=300, channels='BGR', clamp=True, caption="Result Image")

    '''Display Image Test Result with GPEN'''
    image_orig_col_gpen, image_result_col_gpen = st.columns(2)
    with image_orig_col_gpen:
        st.image(content_gpen, width=300, channels='BGR', clamp=True, caption="Original Image with GPEN")
    with image_result_col_gpen:
        st.image(generated_gpen, width=300, channels='BGR', clamp=True, caption="Result Image with GPEN")

    '''Display Time Taken'''
    st.subheader('Time Taken')
    st.markdown("Photo to Cartoon **{} sec**".format(photo_to_cartoon_time))
    st.markdown("GPEN Photo to Cartoon **{} sec**".format(gpen_to_cartoon_time))
    st.markdown("Photo to GPEN Photo **{} sec**".format(photo_to_gpen_time))
    st.markdown("Original Photo Size **{}x{}**".format(orig_w, orig_h))


def webcam_input(gan_type_name, style_model_name, enable_gpen = False):

    st.header("Webcam Live Feed")
    WIDTH = st.select_slider('QUALITY (May reduce the speed)', list(range(150, 501, 50)), value=500)

    if st.checkbox('Enable GPEN'):
        enable_gpen = True
    else:
        enable_gpen = False

    class NeuralStyleTransferTransformer(VideoProcessorBase):
        def __init__(self) -> None:
            self._width = WIDTH

            if gan_type_name == 'AnimeGAN_v2' or gan_type_name == 'AnimeGAN':
                if gan_type_name == 'AnimeGAN_v2':
                    self.style_model_path = animegan_v2_model_dict[style_model_name]
                elif gan_type_name == 'AnimeGAN':
                    self.style_model_path = animegan_model_dict[style_model_name]
                self.model = get_model_from_path(self.style_model_path)

            #elif gan_type_name == 'CutGAN':
            #    self.style_model_path = cutgan_model_dict[style_model_name]

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            if enable_gpen:
                img, orig_faces, enhanced_faces = faceenhancer.process(img)
            orig_h, orig_w = img.shape[0:2]

            if gan_type_name == 'AnimeGAN_v2' or gan_type_name == 'AnimeGAN':
                img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
                img = img[50:-50, 50:-50]
                img = cv2.resize(img, dsize=(720, 720), interpolation=cv2.INTER_LANCZOS4)
                generated = style_transfer(img, self.model)
                generated = cv2.convertScaleAbs(generated, alpha=255.0)

            elif gan_type_name == 'CutGAN':
                img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_LANCZOS4)
                generated = cut_gan.start_converting(img)
                generated = cv2.resize(generated, dsize=(orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
                generated = cv2.cvtColor(generated, cv2.COLOR_RGB2BGR)

            #generated = cv2.resize(generated, dsize=(orig_h, orig_w), interpolation=cv2.INTER_LANCZOS4)

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
