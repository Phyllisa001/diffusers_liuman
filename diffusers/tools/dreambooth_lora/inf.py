import torch
from diffusers import StableDiffusionPipeline
import os


# prompts = ["a photo of haomo-night night traffic scene", "a photo of sks night traffic scene", ]
# model_names = ["haomo_night", "haomo_night_sks", ]

prompts = ["a photo of sks night traffic scene", ]
model_names = [ "haomo_night_sks_2", ]

model_base = "/mnt/share_disk/lei/git/diffusers/local_models/stable-diffusion-v1-5"
n = 6

for ind, model_name in enumerate(model_names):
    model_path = "./res/finetune/dreambooth_lora/%s" % model_name
    res_dir = "./vis/dreambooth_lora/%s" % model_name
    os.makedirs(res_dir, exist_ok=True)
    pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
    pipe.unet.load_attn_procs(model_path)
    pipe.to("cuda")

    prompt = prompts[ind]
    for i in range(n):
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, cross_attention_kwargs={"scale": 1},).images[0]
        res_id = "%s/%s_%d.png" % (res_dir, "_".join(prompt.split(" ")), i)
        image.save(res_id)
        print(res_id)