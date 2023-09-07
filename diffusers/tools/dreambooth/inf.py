# from diffusers import DiffusionPipeline, UNet2DConditionModel
# from transformers import CLIPTextModel
# import torch

# # Load the pipeline with the same arguments (model, revision) that were used for training
# model_id = "CompVis/stable-diffusion-v1-4"

# unet = UNet2DConditionModel.from_pretrained("/sddata/dreambooth/daruma-v2-1/checkpoint-100/unet")

# # if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
# text_encoder = CLIPTextModel.from_pretrained("/sddata/dreambooth/daruma-v2-1/checkpoint-100/text_encoder")

# pipeline = DiffusionPipeline.from_pretrained(model_id, unet=unet, text_encoder=text_encoder, dtype=torch.float16)
# pipeline.to("cuda")

# # Perform inference, or save, or push to the hub
# pipeline.save_pretrained("dreambooth-pipeline")


from diffusers import DiffusionPipeline, StableDiffusionPipeline
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import cv2
import numpy as np  
from tqdm import tqdm

prompts = ["a street filled with lots of traffic at night time with lights on and cars driving down the street and a building in the background", 
           "a street that covered by heavy snow, filled with lots of traffic and cars driving down the street and a building in the background, 4k",
           "a street filled with lots of traffic and cars driving down the street and a building in the background, 4k",
           "a street filled with lots of traffic and cars driving down the street and a building in the background, rainy, 4k",
           "a photo of a dog",
           "a group of people riding on the back of three wheeled vehicles down a street next to a traffic light",
           "a roundabout filled with lots of cars and trucks in a foggy day, 4k",
           "a street filled with lots of traffic and cars driving down the street, 4k",
           "a city street with cars driving down it and tall buildings in the background on a foggy day with a few cars",
           "a white bus driving down a street next to a white car and a white car with a yellow license plate",
           "two cars collisions in the street",
           "group of people crossing the road",
           "a street covered by heavy snow, filled with lots of traffic and cars driving down the street and The Eiffel Tower in the background, 4k"]
# prompts = ["nighttime, 4k"]



# model_names = ["SD-Base", "SD-HM-V0.0", "SD-HM-V0.1", "SD-HM-V1.0", "SD-HM-V1.1", "SD-HM-V1.2", "SD-HM-V2.0", "SD-HM-V3.0", "SD-HM-V3.0.1", "SD-HM-V3.1", "SD-HM-V3.1.1", "SD-HM-V4.0", "SD-HM-V4.0.1", "SD-HM-V4.1", "SD-HM-V4.1.1"]
model_names = ["SD-HM-V0.6.2"]

# model_dir = "./res/finetune/dreambooth" 
model_dir = "/mnt/ve_share/songyuhao/generation/models/online/diffusions/res/finetune/dreambooth"
n = 20
combine = False

for ind, model_name in enumerate(model_names):
    print(model_name)
    res_dir = "/mnt/ve_share/songyuhao/generation/data/result/diffusions/vis/dreambooth/%s" % model_name
    os.makedirs(res_dir, exist_ok=True)
    
    if model_name == "SD-Base":
        model_id = "/mnt/ve_share/songyuhao/generation/models/online/diffusions/base/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    
    else:
        model_id= "%s/%s" % (model_dir, model_name)
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    
    for prompt in prompts:
        print(prompt)
        img_lst = []
        if model_name == "SD-HM-V0.1.0":
            prompt += ", in the style of haomo"
        res_dir_p = "%s/%s" % (res_dir, "_".join(prompt.split(" ")))
        os.makedirs(res_dir_p, exist_ok=True)
        
        file_count = 0
        for _, _, files in os.walk(res_dir_p):
            file_count += len(files)
        
        if file_count >= n:        
            continue
        
        for i in tqdm(range(n)):
            image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
            res_id = "%s/%d.png" % (res_dir_p, i)
            image.save(res_id)
                
                
                
                
                
        #     if combine:
        #         image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) 
        #         img_lst.append(image)

        # if combine:
        #     list_of_lists = [img_lst[i : i + 4] for i in range(0, len(img_lst), 4)]
        #     im_combined = cv2.vconcat([cv2.hconcat(_) for _ in list_of_lists])
        #     res_id = "%s/combined.png" % (res_dir_p)
        #     cv2.imwrite(res_id, im_combined)
        #     print(res_id)