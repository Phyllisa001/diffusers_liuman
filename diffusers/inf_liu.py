from diffusers import StableDiffusionPipelineLiu, StableDiffusionPipeline
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import cv2
import numpy as np  
from tqdm import tqdm

prompts = ["the time is about 23:00, a street filled with lots of traffic and cars driving down the street and a building in the background", 
           "the time is about 23:00, a street filled with lots of traffic and cars driving down the street and a building in the background",
           "the time is about 23:00, a street filled with lots of traffic and cars driving down the street and a building in the background",
           "the time is about 23:00, a street filled with lots of traffic and cars driving down the street and a building in the background",
           "the time is about 23:00, a street filled with lots of traffic and cars driving down the street and a building in the background",
           "the time is about 23:00, a street filled with lots of traffic and cars driving down the street and a building in the background",
           "the time is about 23:00, a street filled with lots of traffic and cars driving down the street and a building in the background",
           "the time is about 23:00, a street filled with lots of traffic and cars driving down the street and a building in the background",
           "the time is about 23:00, a street filled with lots of traffic and cars driving down the street and a building in the background",
           "the time is about 23:00, a street filled with lots of traffic and cars driving down the street and a building in the background"]

pipe = StableDiffusionPipelineLiu.from_pretrained("/mnt/ve_share/liuman/model/SD-HMft0.8.2_withtime", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
output_dir = "/mnt/share_disk/liuman/diffusers/inf_images/SD-HMft0.8.2_withtime/time23"

    
for i, prompt in enumerate(prompts):
    image = pipe(prompt).images[0]
    # 保存生成的图片
    image_path = os.path.join(output_dir, f"image_{i}.png")
    image.save(image_path)