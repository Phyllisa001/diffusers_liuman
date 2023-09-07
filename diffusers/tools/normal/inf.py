import torch
from diffusers import StableDiffusionPipeline
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import cv2
import numpy as np  
from tqdm import tqdm
pipe = StableDiffusionPipeline.from_pretrained("/mnt/share_disk/liuman/SD-HM/SD-HM-V0.4.0", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
prompt = "the time is about 10:00pm, a street filled with lots of traffic and cars driving down the street and a building in the background"
image = pipe(prompt).images[0]


