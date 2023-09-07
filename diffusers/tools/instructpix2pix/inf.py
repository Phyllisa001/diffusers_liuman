import PIL
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, UniPCMultistepScheduler, UNet2DConditionModel
import os
import cv2
from tqdm import tqdm
import numpy as np
from transformers import CLIPTextModel



def preprocess_image(url):
    # image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.Image.open(url)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize((512, 512))
    # image = image.resize((512, 512))
    
    return image

# prompts = ["make it dawn", "make it dusk", "make it night", "make it rainy", "make it snowy", "make it cloudy", "make it foggy", "make it contre-jour", "make it backlight"]
prompts = ["make it snowy"]

# "INS-HM-SNOWY-V0.3.0", "INS-HM-SNOWY-V0.3.0/checkpoint-2500", "INS-HM-SNOWY-V0.3.0/checkpoint-5000", "INS-HM-SNOWY-V0.3.0/checkpoint-7500", "INS-HM-SNOWY-V0.3.0/checkpoint-10000", "INS-HM-SNOWY-V0.3.0/checkpoint-12500"
model_names = [
               "INS-HM-SNOWY-V0.4.0", "INS-HM-SNOWY-V0.4.0/checkpoint-2500", "INS-HM-SNOWY-V0.4.0/checkpoint-5000", "INS-HM-SNOWY-V0.4.0/checkpoint-7500", "INS-HM-SNOWY-V0.4.0/checkpoint-10000", "INS-HM-SNOWY-V0.4.0/checkpoint-12500"]

# model_dir = "/mnt/ve_share/songyuhao/generation/models/online/diffusions/res/instructpix2pix/model"
model_dir = "/mnt/share_disk/songyuhao/models/online/diffusions/res/instructpix2pix/model"

combine = True

test_path = '/mnt/ve_share/songyuhao/generation/data/test/v0.0'
res_root = "/mnt/ve_share/songyuhao/generation/data/result/diffusions/vis/instructpix2pix/official"

image_paths = []
for foldername, subfolders, filenames in os.walk(test_path):
    for filename in filenames:
        # Get the full path of the file
        file_path = os.path.join(foldername, filename)
        image_paths.append(file_path)
        
n = len(image_paths)
print(n)

for ind, model_name in enumerate(model_names):
    print(model_name)
    res_dir = "%s/%s" % (res_root, model_name.split("/")[0] + "-" + model_name.split("/")[-1].split("-")[-1]) if "/" in model_name else "%s/%s" % (res_root, model_name)
    os.makedirs(res_dir, exist_ok=True)
    
    if model_name == "INS-Base":
        model_id = "/mnt/ve_share/songyuhao/generation/models/online/diffusions/base/instruct-pix2pix"
    else:
        model_id= "%s/%s" % (model_dir, model_name)
        
    if "/" in model_name:
        unet = UNet2DConditionModel.from_pretrained("%s/unet_ema" % model_id)
        model_id_true = "/".join(model_id.split("/")[:-1])
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id_true, unet=unet).to("cuda")
        
    else:
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    
    for prompt in prompts:
        print(prompt)
        img_lst = []
        res_dir_p = "%s/%s" % (res_dir, "_".join(prompt.split(" ")))
        os.makedirs(res_dir_p, exist_ok=True)
        
        file_count = 0
        for _, _, files in os.walk(res_dir_p):
            file_count += len(files)
        
        if file_count >= n:        
            continue
        
        for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
            test_image = preprocess_image(image_path)
            image = pipe(prompt, image=test_image, num_inference_steps=50, image_guidance_scale=1.5, guidance_scale=7).images[0]
            res_id = "%s/%d.png" % (res_dir_p, i)
            
            if combine:
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) 
                test_image = cv2.cvtColor(np.array(test_image), cv2.COLOR_RGB2BGR) 
                im_combined = cv2.hconcat([test_image, image])
                cv2.imwrite(res_id, im_combined)

            else:
                image.save(res_id)
                