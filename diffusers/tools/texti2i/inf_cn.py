import PIL
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, UniPCMultistepScheduler, UNet2DConditionModel, ControlNetModel
from diffusers.pipelines.controlnet import StableDiffusionControlNetImg2ImgPipeline
import os
import cv2
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
import numpy as np
from transformers import CLIPTextModel
from diffusers.utils import load_image

def preprocess_image(url):
    # image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.Image.open(url)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize((512, 512))
    # image = image.resize((512, 512))
    
    return image


def canny(url):
    image = np.array(load_image(url))

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image
    
    
controlnet_path_canny = "/mnt/ve_share/songyuhao/generation/models/online/diffusions/base/control_v11p_sd15_canny"
controlnet = ControlNetModel.from_pretrained(controlnet_path_canny, torch_dtype=torch.float16).to("cuda")

contorlnet_scale = 0.1
    
prompts = ["make it night", "make it rainy", "make it snowy"]


model_names = ["INS-HM-V0.1.0"]
model_dir = "/mnt/ve_share/songyuhao/generation/models/online/diffusions/res/instructpix2pix/"
combine = True

test_path = '/mnt/ve_share/songyuhao/generation/data/test/v0.0'
res_root = "/mnt/ve_share/songyuhao/generation/data/result/diffusions/vis/instructpix2pix/official_test"

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
        model_id_true = "/mnt/ve_share/songyuhao/generation/models/online/diffusions/base/stable-diffusion-v1-5"
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(model_id_true, controlnet=controlnet, torch_dtype=torch.float16).to("cuda")
        
    else:
        model_id = "/mnt/ve_share/songyuhao/generation/models/online/diffusions/base/stable-diffusion-v1-5"
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(model_id, controlnet=controlnet, torch_dtype=torch.float16).to("cuda")
    
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
            control_image = canny(image_path)
            image = pipe(prompt=prompt, image=test_image, control_image=control_image, num_inference_steps=50).images[0]
            res_id = "%s/%d.png" % (res_dir_p, i)
            
            if combine:
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) 
                test_image = cv2.cvtColor(np.array(test_image), cv2.COLOR_RGB2BGR) 
                im_combined = cv2.hconcat([test_image, image])
                cv2.imwrite(res_id, im_combined)

            else:
                image.save(res_id)
                