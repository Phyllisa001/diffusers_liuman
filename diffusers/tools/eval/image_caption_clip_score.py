from diffusers import DiffusionPipeline, StableDiffusionPipeline
import torch
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import cv2
from PIL import Image
import os
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import datetime


style = "load"  # ["inf" | "load"]
# model_names = ["SD-Base", "SD-HM-V0.0", "SD-HM-V0.1", "SD-HM-V1.0", "SD-HM-V1.1", "SD-HM-V1.2", "SD-HM-V2.0", "SD-HM-V3.0", "SD-HM-V3.0.1", "SD-HM-V3.1", "SD-HM-V4.0", "SD-HM-V4.1"]
model_names = ["SD-Base", "SD-HM-V0.0", "SD-HM-V0.1", "SD-HM-V1.0", "SD-HM-V1.1", "SD-HM-V1.2", "SD-HM-V2.0", "SD-HM-V3.0", "SD-HM-V3.0.1", "SD-HM-V3.1", "SD-HM-V3.1.1", "SD-HM-V4.0", "SD-HM-V4.0.1", "SD-HM-V4.1", "SD-HM-V4.1.1"]

model_dir = "/mnt/ve_share/songyuhao/generation/models/online/diffusions/res/finetune/dreambooth"
clip_path = "/mnt/ve_share/songyuhao/generation/models/online/diffusions/base/clip-vit-base-patch16"
clip_score_fn = partial(clip_score, model_name_or_path=clip_path)

def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(folder_path, filename)
            image = Image.open(file_path)
            if image is not None:
                images.append(np.array(image))
    return images


if style == "inf":
    prompts = [
        "a photo of an astronaut riding a horse on mars",
        "A high tech solarpunk utopia in the Amazon rainforest",
        "A pikachu fine dining with a view to the Eiffel Tower",
        "A mecha robot in a favela in expressionist style",
        "an insect robot preparing a delicious meal",
        "A small cabin on top of a snowy mountain in the style of Disney, artstation",
    ]

    seed = 0
    generator = torch.manual_seed(seed)

    model_ckpt = "/mnt/ve_share/songyuhao/generation/models/online/diffusions/base/stable-diffusion-v1-5"
    sd_pipeline = StableDiffusionPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16).to("cuda")
    images = sd_pipeline(prompts, num_images_per_prompt=3, generator=generator, output_type="np").images
    print(images.shape)
    clip_score = calculate_clip_score(images, prompts)
    
elif style == "load":
    res = dict()
    clip_scores = []
    for ind, model_name in enumerate(tqdm(model_names)):
        model_dir = "/mnt/ve_share/songyuhao/generation/data/result/diffusions/vis/dreambooth/%s" % model_name
        images, prompts = [], [] 
        for prompt in os.listdir(model_dir):
            prompt_dir = "%s/%s" % (model_dir, prompt)
            prompt_str = prompt.replace("_", " ")
            images_p, prompts_p = [], []
            for image_p in os.listdir(prompt_dir):
                image_path = os.path.join(prompt_dir, image_p)
                images_p.append(np.array(Image.open(image_path)))
                prompts_p.append(prompt_str)
            images_p = random.choices(images_p, k=8)
            prompts_p = random.choices(prompts_p, k=8)
            images += images_p
            prompts += prompts_p
        # prompts = np.array(prompts)
        images = np.array(images)
        clip_score = calculate_clip_score(images, prompts)
        print(model_name, clip_score)
        clip_scores.append(clip_score)
        
        
    res = {"Model": model_names,
           "CLIP_Score": clip_scores}
    res_pd = pd.DataFrame(res)
    
    current_datetime = datetime.datetime.now()

    # Format the date and time as a string
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = "/mnt/ve_share/songyuhao/generation/data/result/diffusions/eval/%s.xlsx" % formatted_datetime
    print(save_path)
    res_pd.to_excel(save_path, index=True)
