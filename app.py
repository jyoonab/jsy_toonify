import streamlit as st
import cv2
import av
import numpy as np
import threading

from PIL import Image
from data.data_list import *
from streamlit_webrtc import webrtc_streamer, ClientSettings, WebRtcMode, VideoProcessorBase
from image_preprocessor.FaceCropper import FaceCropper
from ModelManager import Vid2vid, AnimeGan, CUT
from summary_page import project_summary_page

from image_preprocessor.super_resolution.real_esrnet import RealESRNet

def cartoonizer_demo_page(filter: str, cartoonizer_mode: str) -> None:
    st.header("Webcam Live Test")

    class WebcamProcessor(VideoProcessorBase):
        def __init__(self) -> None:
            self.vid2vid = None
            self.face_detector = None
            self.anime_gan = None
            self.cut = None
            self.super_resolution = None

            self.target_image = []
            self.counter = 0

            self.filter = filter
            self.cartoonizer_mode = cartoonizer_mode

            t = threading.Thread(target=self.start_models)
            t.start()

        def recv(self, frame: np.ndarray) -> av.VideoFrame:
            self.counter = self.counter + 1

            frame = frame.to_ndarray(format="rgb24")

            #If Model is not loaded yet
            if not self.is_model_loading_finished():
                return av.VideoFrame.from_ndarray(frame)

            #If Vid2vid
            if self.cartoonizer_mode == "Vid2vid":
                self.face_detector.get_facial_landmarks(frame)
                cropped_frame_int = self.face_detector.crop_image(frame) # Crop the Image by Face
                cropped_frame_float = cropped_frame_int.astype(np.float32)/255

                if self.counter < 10:
                    return av.VideoFrame.from_ndarray(frame)

                elif self.filter == "Self" and len(self.target_image) == 0:
                    self.target_image = self.anime_gan.start_converting(cropped_frame_int, self.filter)
                    self.vid2vid.change_target_image_from_array(self.target_image)

                if self.filter == "Hand Drawing":
                    self.vid2vid.change_target_image_from_image("./models/face_vid2vid/asset/avatar/3.png")

                if self.filter == "Aman":
                    self.vid2vid.change_target_image_from_image("./models/face_vid2vid/asset/avatar/6.png")

                result_frame = self.vid2vid.start_converting(cropped_frame_float)
            #If CUT
            elif self.cartoonizer_mode == "Selfie Segmentation":
                result_frame = self.cut.start_converting_background_only(frame, self.filter)

            #If Anime_Gan
            elif self.cartoonizer_mode == "Cartoonizer":
                result_frame = self.anime_gan.start_converting(frame, self.filter)

            if result_frame.dtype != "uint8":
                result_frame = np.round(255 * result_frame).astype(np.uint8)
            result_frame = cv2.resize(result_frame, dsize=(720, 720), interpolation=cv2.INTER_LANCZOS4)
            frame = cv2.resize(frame, dsize=(720, 720), interpolation=cv2.INTER_LANCZOS4)

            img_list = [frame, result_frame]
            combined_frame = cv2.hconcat(img_list)

            return av.VideoFrame.from_ndarray(combined_frame)

        def is_model_loading_finished(self) -> bool:
            if self.vid2vid is None or self.face_detector is None or self.anime_gan is None or self.cut is None or self.super_resolution is None:
                return False
            return True

        def start_models(self) -> None:
            self.vid2vid = Vid2vid()
            self.face_detector = FaceCropper()
            self.anime_gan = AnimeGan()
            self.cut = CUT()
            self.super_resolution = RealESRNet()

        def update_model_name(self, filter: str, cartoonizer_mode: str) -> None:
            if self.cartoonizer_mode != cartoonizer_mode or self.filter != filter:
                self._update_model(filter, cartoonizer_mode)

        def _update_model(self, filter: str, cartoonizer_mode: str) -> None:
            self.filter = filter
            self.cartoonizer_mode = cartoonizer_mode

    ctx = webrtc_streamer(
        rtc_configuration=ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        ),
        media_stream_constraints={"video": True, "audio": False},
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=WebcamProcessor,
        key="result-stream",
    )

    if ctx.video_transformer:
        ctx.video_transformer.update_model_name(filter, cartoonizer_mode)


st.title("JSY's Cartoonizer Demo Page")
main_logo = Image.open('src/logo.png')
st.sidebar.image(main_logo, use_column_width=True, width=50)
st.sidebar.title('Menu')
method = st.sidebar.radio('', options=['About Projects', 'Demo Page'])

if method == 'About Projects':
    project_summary_page()

if method == 'Demo Page':
    st.sidebar.title('Video Filter Options')

    cartoonizer_mode = st.sidebar.selectbox("Choose the cartoonizer mode: ", mode_list)
    if cartoonizer_mode == 'Selfie Segmentation':
        filter = st.sidebar.selectbox("Choose the video filter: ", background_list)
    elif cartoonizer_mode == 'Vid2vid':
        filter = st.sidebar.selectbox("Choose the video filter: ", vid2vid_image_list)
    else:
        filter = st.sidebar.selectbox("Choose the video filter: ", filter_name)

    cartoonizer_demo_page(filter, cartoonizer_mode)
