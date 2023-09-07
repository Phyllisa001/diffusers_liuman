import torch
import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline

device = "cuda"

model_id = "./res/finetune/dreambooth/haomo_night"
image_path = "/mnt/ve_share/songyuhao/generation/data/train/GAN/night/trainA/29156764284920349.jpg"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
init_image = Image.open(image_path).convert("RGB").resize([768, 512])

prompt = "haomo-night"
generator = torch.Generator(device=device).manual_seed(1024)
image = pipe(prompt=prompt, image=init_image, strength=0.7, guidance_scale=5, generator=generator).images[0]
image.save("hahah.png")


