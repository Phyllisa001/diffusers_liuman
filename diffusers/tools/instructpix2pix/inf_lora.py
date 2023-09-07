import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import PIL
from diffusers import StableDiffusionInstructPix2PixPipeline, UniPCMultistepScheduler, UNet2DConditionModel
import cv2
from tqdm import tqdm
import numpy as np
from transformers import CLIPTextModel
import torch


def preprocess_image(url):
    # image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.Image.open(url)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize((512, 512))
    # image = image.resize((512, 512))
    
    return image

model_base = "/mnt/ve_share/songyuhao/generation/models/online/diffusions/base/instruct-pix2pix"
prompts = ["make it night"]
lora_names = ["INS-HM-LORA-NIGHT-V0.3.0", "INS-HM-LORA-NIGHT-V0.3.0/checkpoint-5000", "INS-HM-LORA-NIGHT-V0.3.0/checkpoint-10000", "INS-HM-LORA-NIGHT-V0.3.0/checkpoint-15000", ]
            #   "INS-HM-LORA-NIGHT-V0.1.2",  "INS-HM-LORA-NIGHT-V0.1.2/checkpoint-5000", "INS-HM-LORA-NIGHT-V0.1.2/checkpoint-10000", "INS-HM-LORA-NIGHT-V0.1.2/checkpoint-15000", 
            #   "INS-HM-LORA-NIGHT-V0.1.3",  "INS-HM-LORA-NIGHT-V0.1.3/checkpoint-5000", "INS-HM-LORA-NIGHT-V0.1.3/checkpoint-10000", "INS-HM-LORA-NIGHT-V0.1.3/checkpoint-15000", 
            #   "INS-HM-LORA-NIGHT-V0.1.4",  "INS-HM-LORA-NIGHT-V0.1.4/checkpoint-5000", "INS-HM-LORA-NIGHT-V0.1.4/checkpoint-10000", "INS-HM-LORA-NIGHT-V0.1.4/checkpoint-15000", ]
model_dir = "/mnt/ve_share/songyuhao/generation/models/online/diffusions/res/instructpix2pix/lora"
combine = True

test_path = '/mnt/ve_share/songyuhao/generation/data/test/v0.0'
res_root = "/mnt/ve_share/songyuhao/generation/data/result/diffusions/vis/instructpix2pix/lora"

# test_path = '/mnt/ve_share/songyuhao/generation/data/test/kl/'
# res_root = "/mnt/ve_share/songyuhao/generation/data/result/diffusions/vis/instructpix2pix/casual"

image_paths = []
for foldername, subfolders, filenames in os.walk(test_path):
    for filename in filenames:
        # Get the full path of the file
        file_path = os.path.join(foldername, filename)
        image_paths.append(file_path)
        
n = len(image_paths)
print(n)

for ind, lora_name in enumerate(lora_names):
    print(lora_name)
    res_dir = "%s/%s" % (res_root, lora_name.split("/")[0] + "-" + lora_name.split("/")[-1].split("-")[-1]) if "/" in lora_name else "%s/%s" % (res_root, lora_name)
    os.makedirs(res_dir, exist_ok=True)
    
    
    
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_base, torch_dtype=torch.float16).to("cuda")
    if lora_name == "INS-Base":
        pass
    # elif "/" in lora_name:
    #     lora_model = "%s/%s" % (model_dir, lora_name)
    #     pipe.unet.load_attn_procs(lora_model)
    #     pipe.to("cuda")
    else:
        lora_model = "%s/%s" % (model_dir, lora_name)
        pipe.unet.load_attn_procs(lora_model)
        pipe.to("cuda")
        # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
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
                