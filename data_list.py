import os

# GAN Type
GAN_type_name = ['AnimeGAN_v2', 'AnimeGAN', 'CutGAN', 'SytleGAN']

# AnimeGAN
animegan_v2_file_list = ['portrait_v2_512.pt']
animegan_v2_type_list = ['Portrait']
model_path = 'models/animeGAN_v2'
animegan_v2_model_dict = {name: os.path.join(model_path, filee) for name, filee in zip(animegan_v2_type_list, animegan_v2_file_list)}

# AnimeGAN
animegan_file_list = ['portrait.pt', 'weater_and_yourname.pt', 'face_paint_512_v2_0.pt', 'test.pt', 'metfaces_512_v1_0.pt']
animegan_type_list = ['Portrait', 'Your Name', 'Face Paint', 'Without Mean', 'Metfaces']
model_path = 'models/animeGAN'
animegan_model_dict = {name: os.path.join(model_path, filee) for name, filee in zip(animegan_type_list, animegan_file_list)}

# CutGAN
cutgan_file_list = ['latest_net_G.pth']
cutgan_type_list = ['FastCUT']
model_path = 'models/cutGAN'
cutgan_model_dict = {name: os.path.join(model_path, filee) for name, filee in zip(cutgan_type_list, cutgan_file_list)}

# Style Images Data
content_images_file = ['iu.jpg', 'ancient_city.jpg', 'blue-moon-lake.jpg', 'Dawn Sky.jpg', 'Dipping-Sun.jpg', 'golden_gate.jpg', 'Japanese-cherry.jpg', 'jurassic_park.jpg', 'Kinkaku-ji.jpg', 'messi.jpg', 'sagano_bamboo_forest.jpg', 'Sunlit Mountains.jpg', 'tubingen.jpg', 'winter-wolf.jpg']
content_images_name = ['IU', 'Ancient_city', 'Blue-moon-lake', 'Dawn sky', 'Dipping-sun', 'Golden_gate', 'Japanese-cherry', 'Jurassic_park', 'Kinkaku-ji', 'Messi', 'Sagano_bamboo_forest', 'Sunlit mountains', 'Tubingen', 'Winter-wolf']
images_path = 'images'
content_images_dict = {name: os.path.join(images_path, filee) for name, filee in zip(content_images_name, content_images_file)}
