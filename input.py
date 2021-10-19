import threading
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import imutils
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings
from neural_style_transfer import get_model_from_path, style_transfer, image2tensor, tensor2image, imshow, generate_new_video
from data import *
from torchvision import transforms
import tempfile


def image_input(style_model_name):
    style_model_path = style_models_dict[style_model_name]
    model = get_model_from_path(style_model_path)

    # Upload or Image_Select
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

    WIDTH = st.sidebar.select_slider('QUALITY (May reduce the speed)', list(range(150, 501, 50)), value=200)
    content = imutils.resize(content, width=WIDTH)
    generated = style_transfer(content, model)


    st.sidebar.image(content, width=300, channels='BGR')
    st.image(generated, channels='RGB', clamp=True, caption="Result Image")
    #cv2.imwrite('./result/face_results.jpg', cv2.cvtColor(255*tensor2image(generated), cv2.COLOR_BGR2RGB))

def video_input(style_model_name):
    converting = False

    style_model_path = style_models_dict[style_model_name]
    model = get_model_from_path(style_model_path)

    video_file = st.file_uploader('Which video?', type=['mp4', 'avi'])
    if video_file:
        # show source video
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(video_file.read())
        st.video(temp_file.name)


        if converting:
            st.write('Converting...')
        else:
            if st.button('Start Converting'):
                converting = True
                st.video(generate_new_video(temp_file.name, model))


def webcam_input(style_model_name):
    st.header("Webcam Live Feed")
    WIDTH = st.sidebar.select_slider('QUALITY (May reduce the speed)', list(range(150, 501, 50)))

    class NeuralStyleTransferTransformer(VideoTransformerBase):
        _width = WIDTH
        _model_name = style_model_name
        _model = None

        def __init__(self) -> None:
            self._model_lock = threading.Lock()

            self._width = WIDTH
            self._update_model()

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
            style_model_path = style_models_dict[self._model_name]
            with self._model_lock:
                self._model = get_model_from_path(style_model_path)

        def transform(self, frame):
            image = frame.to_ndarray(format="bgr24")

            if self._model == None:
                return image

            orig_h, orig_w = image.shape[0:2]

            # cv2.resize used in a forked thread may cause memory leaks
            input = np.asarray(Image.fromarray(image).resize((self._width, int(self._width * orig_h / orig_w))))

            with self._model_lock:
                transferred = style_transfer(input, self._model)

            result = Image.fromarray((transferred * 255).astype(np.uint8))
            return np.asarray(result.resize((orig_w, orig_h)))


    ctx = webrtc_streamer(
        client_settings=ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        ),
        video_transformer_factory=NeuralStyleTransferTransformer,
        key="neural-style-transfer",
    )
    if ctx.video_transformer:
        ctx.video_transformer.set_width(WIDTH)
        ctx.video_transformer.update_model_name(style_model_name)
