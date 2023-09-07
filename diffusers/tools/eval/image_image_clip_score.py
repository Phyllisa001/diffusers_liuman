from PIL import Image
import requests
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
import torch
import numpy as np
from numpy.linalg import norm

# Load the CLIP model
clip_path = "/mnt/ve_share/songyuhao/generation/models/online/diffusions/base/clip-vit-base-patch16"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your dataset of image pairs
image_paths_1 = [...]  # List of paths to the first images
image_paths_2 = [...]  # List of paths to the second images

# Preprocess the images
images_1 = ["/tos://haomo-public/lucas-generation/syh/train/instructpix2pix/imgs/replace_blend_reweight/night/0/0.80_0.80_2.00/a_car_driving_down_a_highway_at_daytime_with_mountains_in_the_background_and_a_bridge_in_the_foreground_with_a_car_driving_on_the_road_at_daytime.png"]
images_2 = ["/tos://haomo-public/lucas-generation/syh/train/instructpix2pix/imgs/replace_blend_reweight/night/0/0.80_0.80_2.00/a_car_driving_down_a_highway_at_nighttime_with_mountains_in_the_background_and_a_bridge_in_the_foreground_with_a_car_driving_on_the_road_at_nighttime.png"]

with torch.no_grad():
    model = CLIPModel.from_pretrained(clip_path)
    processor = AutoProcessor.from_pretrained(clip_path)

    for i in range(len(images_1)):
        image_1 = Image.open(images_1[i])
        image_2 = Image.open(images_2[i])

        input_1 = processor(images=image_1, return_tensors="pt")
        image_feature_1 = model.get_image_features(**input_1).view(-1)
        print(image_feature_1.shape) 
        print(image_feature_1) 
        
        input_2 = processor(images=image_2, return_tensors="pt")
        image_feature_2 = model.get_image_features(**input_2).view(-1)
        print(image_feature_2.shape) 
        print(image_feature_2) 
        
        sim = round(np.dot(image_feature_1, image_feature_2) / (norm(image_feature_1) * norm(image_feature_2)), 4)
        print(sim)

