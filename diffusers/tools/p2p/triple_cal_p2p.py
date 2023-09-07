from PIL import Image
import os 
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
import torch
import numpy as np
from numpy.linalg import norm
import json
import torch.nn.functional as F
from tqdm import tqdm

clip_path = "/mnt/ve_share/songyuhao/generation/models/online/diffusions/base/clip-vit-base-patch16"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imgs_root = "/mnt/ve_share/songyuhao/generation/data/p2p_cn/imgs/exp2"

save_json_path = "/mnt/ve_share/songyuhao/generation/data/p2p_exp/exp2.json"

model = CLIPModel.from_pretrained(clip_path)
processor = AutoProcessor.from_pretrained(clip_path)
tokenizer = AutoTokenizer.from_pretrained(clip_path)

def text_feature_extractor(prompt):
    text_token = tokenizer([prompt], padding=True, return_tensors="pt")
    text_feature = model.get_text_features(**text_token).view(-1)
    return text_feature

def image_feature_extractor(image_path):
    image = Image.open(image_path)
    image_input = processor(images=image, return_tensors="pt")
    image_feature = model.get_image_features(**image_input).view(-1)
    return image_feature

def cos_sim(fea_1, fea_2):
    return float(round(np.dot(fea_1, fea_2) / (norm(fea_1) * norm(fea_2)), 4))

image_1 = "%s/ori.png" % imgs_root
for root, folders, files in os.walk(imgs_root):
    files.remove("ori.png")
    image_2s = [os.path.join(root, f) for f in files]

prompt_1 = "daytime, a bus and a car on a city street with a bridge in the background and a green car on the road"
prompt_2 = "nighttime, a bus and a car on a city street with a bridge in the background and a green car on the road"

result = []
with torch.no_grad():
    for image_2 in tqdm(image_2s):
        
        text_fea_1 = text_feature_extractor(prompt_1)
        text_fea_2 = text_feature_extractor(prompt_2)
        
        image_fea_1 = image_feature_extractor(image_1)
        image_fea_2 = image_feature_extractor(image_2)
        
        image_image_sim = cos_sim(image_fea_1, image_fea_2)
        image_caption_sim_1 = cos_sim(text_fea_1, image_fea_1)
        image_caption_sim_2 = cos_sim(text_fea_2, image_fea_2)
        directional_sim = cos_sim(image_fea_2 - image_fea_1, text_fea_2 - text_fea_1)
        
        res = {"path": image_2, "image_image_sim": image_image_sim, "image_caption_sim_1": image_caption_sim_1, "image_caption_sim_2": image_caption_sim_2, "directional_sim": directional_sim}
        result.append(res)

result = sorted(result, key=lambda x: x["directional_sim"])
with open(save_json_path, "w") as json_file:
    json.dump(result, json_file)
print(save_json_path)
