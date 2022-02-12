# jsy_toonify

## Summary & Purpose
This demo presents the several methods to create the cartoonizing function in real-time. The main idea is to combine various models to cover each others' weaknesses, so the model can be fast and light enough to be run in real-time without losing any performance.

1. 1st Method : Cartoonize using AnimeGANv2

The first method is to simply put image into the AnimeGANv2 model and receive the result. As the result is shown from the figure from above, the quality of the cartoonized image is high enough. However, the webcam frame test showed the speed of AnimeGANv2 was not fast enough, and I felt little laggy from the webcam output.

![a88aa5ddcbf3a889666ceae70eb71fbb316c445192f0ed7c9d4e5201](https://user-images.githubusercontent.com/37427419/153696538-07ad4a99-c826-4f5b-b46d-537ea5526419.png)

2. 2nd Method : Cartoonize using Vid2vid

The main idea of the second method is to fetch the first webcam frame and use this frame as a source image of Vid2vid. Once the first frame is inserted, it is touched by AnimeGANv2(cartoonizing) and Super Resolution(quality improvement). Once the source image is ready, the Vid2vid generates the video based on the source image.

![835def408ed0917194a8f3c69e52acde3811eb66d771eb72b9fe7b3d](https://user-images.githubusercontent.com/37427419/153696552-415d3dad-e988-4316-9826-548c061e04fd.png)

3. 3rd Method : Cartoonize only background using Selfie Segmentation & CUT

On this method, each webcam frame is transformed by using both Selfie Segmentation and CUT. By using Selfie Segmentation, the background mask will be created. By using CUT, the webcam frame will be cartoonized, and only the background part will be left by cutting the frame based on the background mask.

![79199aa5bedb2e018af197612b77bbca319db9905d1a19979740d689](https://user-images.githubusercontent.com/37427419/153696562-f8fb1800-b9ed-452a-8911-0aa4c9a4191c.png)



## Tech & Libraries
1. Python
2. Streamlit
3. AnimeGANv2
4. Vid2vid
5. MediaPipe
6. CUT

## Requirement
- Python 3.8
- Please see requirements.txt for more required libraries

## Installation & Usage
- Download Vid2vid model from https://drive.google.com/file/d/1lqIzlcfOkd6rXxSSvxHvi1wRp0ap9Uty/view?usp=sharing
- copy this model to ./models/face_vid2vid/ckpt/
```
$ pip install -r requirements.txt
$ streamlit run app.py
```

## Current Status
DONE

## Reference
https://github.com/TachibanaYoshino/AnimeGANv2
https://github.com/NVlabs/imaginaire/tree/master/projects/fs_vid2vid
https://google.github.io/mediapipe/solutions/selfie_segmentation
https://github.com/taesungp/contrastive-unpaired-translation
