import os

# GAN Type
filter_file_list = ['anime_gan/models/face_paint_512_v2_0.pt', 'anime_gan/models/portrait.pt', 'anime_gan/models/weater_and_yourname.pt']
filter_name = ['Face Paint', 'Portrait', 'Your Name']
model_path = 'models'
animegan_model_dict = {name: os.path.join(model_path, filee) for name, filee in zip(filter_name, filter_file_list)}

mode_list = ['Cartoonizer', 'Vid2vid', 'Selfie Segmentation']
background_list = ['Self', 'New York', 'Bang']
vid2vid_image_list = ['Self', 'Hand Drawing', 'Aman']
