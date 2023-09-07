import glob
import os
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image


    
def process(MODEL: str, DEMO: str, COMBINE: bool, FRAME: int = 100) -> None:
    # PATH = "/root/mmgeneration/tests/haomo/%s/%s" % (DEMO, MODEL)
    # PATH = "/mnt/ve_share/songyuhao/generation/data/%s/%s" % (DEMO, MODEL)
    FATHER_PATH = "/mnt/ve_share/songyuhao/generation/data/result/diffusions/vis/dreambooth/%s" % MODEL
    PATH = "%s/%s" %(FATHER_PATH, DEMO)
    img_paths = glob.glob(os.path.join(PATH,'*.png'))
    img_paths = sorted(img_paths)[:FRAME]
    print(len(img_paths))
    
    # if COMBINE:
    #     ORI_PATH = "/mnt/ve_share/songyuhao/generation/data/result/%s/%s_latest/images/real_A" % (MODEL, DEMO)
    #     ori_img_paths = glob.glob(os.path.join(ORI_PATH,'*.png'))
    #     ori_img_paths = sorted(ori_img_paths)[:FRAME]

    # print(img_paths)
    gif_images_ori, gif_images_low = [], []
    
    
    for i, path in tqdm(enumerate(img_paths), total=len(img_paths)):
        # if COMBINE:
        #     im_combine = cv2.hconcat([imageio.imread(ori_img_paths[i]), imageio.imread(path)])
        #     gif_images_ori.append(im_combine)
        #     # gif_images_low.append(resize(im_combine,(1080//6, 1920//3)))
        #     gif_images_low.append(resize(im_combine,(1080//4, 1920//2)))
            
        # else:
        # gif_images_ori.append(imageio.imread(path))
        gif_images_ori.append(Image.open(path))
        
        # gif_images_low.append(resize(imageio.imread(path),(1080//6, 1920//3)))
            
    # save_path = "/root/mmgeneration/tests/haomo/video/ori/%s_%d_ORI_%s.gif" % (SCENE, FRAME, MODEL)
    # save_path = "/mnt/ve_share/songyuhao/generation/data/video/ori/%s/%s_%s_%d_ORI_%s.gif" % (DEMO, DEMO, SCENE, FRAME, MODEL)

    # imageio.mimsave(save_path, gif_images_ori, fps=10)
    # print(save_path)

    # save_path = "/root/mmgeneration/tests/haomo/video/low/%s/%s_%s_%d_LOW_%s.gif" % (DEMO, DEMO, SCENE, FRAME, MODEL)
    save_root = "/mnt/ve_share/songyuhao/generation/data/result/diffusions/vis/dreambooth/gif/%s" % MODEL
    os.makedirs(save_root, exist_ok=True)
    save_path = "%s/%s_%d.gif" % (save_root, DEMO, FRAME)

    # imageio.mimsave(save_path, gif_images_ori, duration=2000)
    gif_images_ori[0].save(save_path, format='GIF', append_images=gif_images_ori[1:], save_all=True, duration=800, loop=0)
    # from pygifsicle import optimize
    # optimize(save_path) # For overwriting the original one
    print(save_path)
    
# "SNOW_BDDCADC", "SNOW_BDDWATERLOO", "SNOW_BDDWATERLOOHAK", "SNOW_HAK"
MODELS = ["SD-Base", "SD-HM-V0.0", "SD-HM-V0.1",]  #  "cut_snow_hok", "cut_snow_waterloo"
# MODELS = ["SD-HM-V1.0", "SD-HM-V1.1", "SD-HM-V1.2"]


for model in MODELS:
    print(model)
    FATHER_PATH = "/mnt/ve_share/songyuhao/generation/data/result/diffusions/vis/dreambooth/%s" % model
    prompts = [x[0].split("/")[-1] for x in os.walk(FATHER_PATH)][1:]
    prompts = ["_".join("a city street with cars driving down it and tall buildings in the background on a foggy day with a few cars".split())]
    print(prompts)
    for prompt in prompts:
        if model == "SD-HM-V0.1":
            prompt += "_,_in_the_style_of_haomo"
        process(MODEL=model, DEMO=prompt, COMBINE=False, FRAME=20)