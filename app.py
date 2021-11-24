import streamlit as st

from data import *
from panel import assignment_one_page, image_input

st.title("JSY's Style Transfer Demo Page")
st.sidebar.title('Style Transfer Options')
method = st.sidebar.radio('', options=['Assignment 1', 'Assignment 2'])

if method == 'Assignment 1':
    st.sidebar.title('Style Model Options')
    convert_target = st.sidebar.radio('Choose the transfer target', options=['Image', 'Webcam'])
    gan_type_name = st.sidebar.selectbox("Choose the style model: ", GAN_type_name)
    if gan_type_name == 'AnimeGAN':
        style_model_name = st.sidebar.selectbox("Choose the style model: ", animegan_type_list)
    elif gan_type_name == 'CutGAN':
        style_model_name = st.sidebar.selectbox("Choose the style model: ", cutgan_type_list)
    assignment_one_page(gan_type_name, convert_target, style_model_name)

elif method == 'Assignment 2':
    st.sidebar.title('Style Model Options')
