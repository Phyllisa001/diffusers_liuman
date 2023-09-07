from diffusers import StableDiffusionPipeline
import torch
import os

prompts = ["A <haomo-night> night traffic scene", ]
model_names = ["haomo_night", ]
model_dir = "./res/finetune/textual_inversion" 

n = 6

for ind, model_name in enumerate(model_names):
    res_dir = "./vis/textual_inversion/%s" % model_name
    os.makedirs(res_dir, exist_ok=True)
    model_id = "%s/%s" % (model_dir, model_name)
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    prompt = prompts[ind]
    
    for i in range(n):
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
        res_id = "%s/%s_%d.png" % (res_dir, "_".join(prompt.split(" ")), i)
        image.save(res_id)
        print(res_id)