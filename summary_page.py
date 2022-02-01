import streamlit as st

def project_summary_page():
    '''Introduction'''
    st.header("Introduction")
    st.write (
        "This demo presents the several methods to create the cartoonizing function in real-time. "
        "The main idea is to combine various models to cover each others' weaknesses, "
        "so the model can be fast and light enough to be run in real-time without losing any performance. "
    )

    '''First Method'''
    st.subheader("1st Method : Cartoonize using AnimeGANv2")
    st.write (
        "AnimeGAN is a model combined between neural style transfer and generative adversarial network(aka GAN), "
        "and this accomplishes the photo transformation from real-world to anime style"
    )
    st.image('./src/anime_gan.png')
    st.write (
        "The first method is to simply put image into the AnimeGANv2 model and receive the result. "
        "As the result is shown from the figure from above, the quality of the cartoonized image is high enough. "
        "However, the webcam frame test showed the speed of AnimeGANv2 was not fast enough, and I felt little laggy from the webcam output. "
    )
    st.image('./src/animeGan_flowchart.png', caption="1st Method Flowchart")
    st.markdown("AnimeGANv2: https://github.com/TachibanaYoshino/AnimeGANv2")

    '''Second Method'''
    st.subheader("2nd Method : Cartoonize using Vid2vid")
    st.write (
        "Vid2vid is a model which does the video-to-video translation between the edge maps and the picture. "
        "By using vid2vid, it is possible to make a picture(Source Image) moving based on the input video (Driving Image). "
    )
    st.image('./src/vid2vid.gif', caption="https://github.com/NVlabs/imaginaire/tree/master/projects/fs_vid2vid")
    st.write (
        "The main idea of the second method is to fetch the first webcam frame and use this frame as a source image of Vid2vid. "
        "Once the first frame is inserted, it is touched by AnimeGANv2(cartoonizing) and Super Resolution(quality improvement). "
        "Once the source image is ready, the Vid2vid generates the video based on the source image. "
    )
    st.image('./src/vid2vid_flowchart.png', caption="2nd Method Flowchart")
    st.markdown("Vid2vid: https://github.com/NVlabs/imaginaire/tree/master/projects/fs_vid2vid")

    st.subheader("3rd Method : Cartoonize only background using Selfie Segmentation & CUT")
    st.write (
        "MediaPipe Selfie Segmentation segments the human shape from the image. "
        "CUT(Contrastive Learning for Unpaired Image-to-Image Translation) translates an image from a source domain to a target domain. "
        "Since I found that the CUT model is fast enough on real-time, and working well on cartoonizing the scenery, I decided to use this model on this method. "
    )
    st.image('./src/cut.png')
    st.write (
        "On this method, each webcam frame is transformed by using both Selfie Segmentation and CUT. "
        "By using Selfie Segmentation, the background mask will be created. "
        "By using CUT, the webcam frame will be cartoonized, and only the background part will be left by cutting the frame based on the background mask. "
    )
    st.image('./src/cut_flowchart.png', caption="3rd Method Flowchart")
    st.markdown("Selfie Segmentation: https://google.github.io/mediapipe/solutions/selfie_segmentation")
    st.markdown("Dataset: https://github.com/taesungp/contrastive-unpaired-translation")
