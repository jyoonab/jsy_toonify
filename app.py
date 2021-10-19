import streamlit as st

from data import *
from input import image_input, video_input, webcam_input

st.title("JSY Style Transfer")
st.sidebar.title('Style Transfer Options')
method = st.sidebar.radio('', options=['Image', 'Video', 'Webcam'])

'''Options Will be Added '''
st.sidebar.title('Style Model Options')
style_model_name = st.sidebar.selectbox("Choose the style model: ", style_models_name)

if method == 'Image':
    image_input(style_model_name)
elif method == 'Video':
    video_input(style_model_name)
else:
    webcam_input(style_model_name)
