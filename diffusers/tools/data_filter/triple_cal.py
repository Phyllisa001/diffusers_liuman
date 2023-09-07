from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
import torch
import numpy as np
from numpy.linalg import norm
import json
import torch.nn.functional as F
from tqdm import tqdm

scene = "snowy"
clip_path = "/mnt/ve_share/songyuhao/generation/models/online/diffusions/base/clip-vit-base-patch16"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = "/mnt/ve_share/songyuhao/generation/data/p2p_cn"
root_json = "%s/ori_jsons/%s_replace.json" % (root, scene) if scene != "snowy" else "%s/ori_jsons/%s_refine_street.json" % (root, scene)
root_img = "replace_blend_reweight/%s" % scene if scene != "snowy" else "refine_blend_reweight/%s" % scene
root_co = "0.70_0.70_2.00" 
length = "all"
save_json_path = "/mnt/ve_share/songyuhao/generation/data/filtered_p2p_cn/ori/%s_%s_%s.json" % (root_img.replace("/", "_"), root_co, length)

with open(root_json, "r") as input_json:
    data = json.load(input_json) if length == "all" else json.load(input_json)[:int(length)]
    
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

result = []
with torch.no_grad():
    for i in tqdm(range(len(data))):
        id_ = data[i]["id"]
        prompt_1 = data[i]["prompt_1"]
        prompt_2 = data[i]["prompt_2"]
        
        image_1 = "%s/imgs/%s/%s/%s/%s.png" % (root, root_img, id_, root_co, "_".join(prompt_1.split(" ")))
        image_2 = "%s/imgs/%s/%s/%s/%s.png" % (root, root_img, id_, root_co, "_".join(prompt_2.split(" ")))

        text_fea_1 = text_feature_extractor(prompt_1)
        text_fea_2 = text_feature_extractor(prompt_2)
        
        image_fea_1 = image_feature_extractor(image_1)
        image_fea_2 = image_feature_extractor(image_2)
        
        image_image_sim = cos_sim(image_fea_1, image_fea_2)
        image_caption_sim_1 = cos_sim(text_fea_1, image_fea_1)
        image_caption_sim_2 = cos_sim(text_fea_2, image_fea_2)
        directional_sim = cos_sim(image_fea_2 - image_fea_1, text_fea_2 - text_fea_1)
        
        res = {"id": id_, "image_image_sim": image_image_sim, "image_caption_sim_1": image_caption_sim_1, "image_caption_sim_2": image_caption_sim_2, "directional_sim": directional_sim}
        result.append(res)
    
    with open(save_json_path, "w") as json_file:
        json.dump(result, json_file)
    # print(save_json_path)
