from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import load_image

image = load_image(
    "/mnt/ve_share/songyuhao/generation/data/test/v0.0/bigcar1.png"
)

import cv2
from PIL import Image
import numpy as np
import torch

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)


from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

controlnet = ControlNetModel.from_pretrained("/mnt/ve_share/songyuhao/generation/models/online/diffusions/base/control_v11p_sd15_canny",)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "/mnt/ve_share/songyuhao/generation/models/online/diffusions/base/stable-diffusion-v1-5", controlnet=controlnet,
)

from diffusers import UniPCMultistepScheduler

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# this command loads the individual model components on GPU on-demand.
pipe.enable_model_cpu_offload()

generator = torch.manual_seed(0)

out_image = pipe(
    "disco dancer with colorful lights", num_inference_steps=20, generator=generator, image=canny_image
).images[0]

out_image.save("./haha.png")