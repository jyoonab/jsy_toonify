import streamlit as st

from data_list import *
from panel import project_summary_page, cartoonizer_demo_page, image_input
from PIL import Image

st.title("JSY's Python Demo Page")
main_logo = Image.open('src/logo4.png')
st.sidebar.image(main_logo, use_column_width=True, width=50)
st.sidebar.title('Menu')
method = st.sidebar.radio('', options=['About Projects', 'Project 1 - Cartoonizer Demo', 'About Me'])

if method == 'About Projects':
    st.sidebar.title('Project 1')
    if st.sidebar.button(" 🦞 Try a Demo at Colab "):
        st.write('Why hello there')
    if st.sidebar.button(" 🍤 See Source Code at Github "):
        st.write('Why hello there')

    project_summary_page()

if method == 'Project 1 - Cartoonizer Demo':
    st.sidebar.title('Style Model Options')
    convert_target = st.sidebar.radio('Choose the transfer target', options=['Image', 'Webcam'])
    gan_type_name = st.sidebar.selectbox("Choose the style model: ", GAN_type_name)
    if gan_type_name == 'AnimeGAN_v2':
        style_model_name = st.sidebar.selectbox("Choose the style model: ", animegan_v2_type_list)
    elif gan_type_name == 'AnimeGAN':
        style_model_name = st.sidebar.selectbox("Choose the style model: ", animegan_type_list)
    elif gan_type_name == 'CutGAN':
        style_model_name = st.sidebar.selectbox("Choose the style model: ", cutgan_type_list)
    cartoonizer_demo_page(gan_type_name, convert_target, style_model_name)
