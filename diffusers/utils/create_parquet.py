import pandas as pd
import random
import os
import json
from tqdm import tqdm

# P2P + ControlNet
P2P_PATH = "/tos://haomo-public/lucas-generation/syh/train/prompt2prompt_controlnet/index.txt"  
# P2P_PATH = "/mnt/ve_share/songyuhao/generation/data/p2p_cn_human.txt"  


# P2P
# P2P_PATH = "tos://haomo-public/lucas-generation/syh/train/instructpix2pix/index.txt"
SCENE = "night"
MODE = "replace_blend_reweight" if SCENE != "snowy" else "refine_blend_reweight"
FOLDER_PATH = "/mnt/ve_share/songyuhao/generation/data/p2p_cn/imgs/%s/%s" % (MODE, SCENE)
TYPE = "folder"
ONLINE = True

PARA = "0.70_0.70_2.00"
SIZE = 500
STREET = False
PARQUET_PATH = "/mnt/ve_share/songyuhao/generation/data/train/diffusions/parquet/%s_%s_%s_%d_street" % (MODE, SCENE, PARA, SIZE) if STREET else "/mnt/ve_share/songyuhao/generation/data/train/diffusions/parquet/%s_%s_%s_%d_cn_filtered" % (MODE, SCENE, PARA, SIZE)
os.makedirs(PARQUET_PATH, exist_ok=True)
PARQUET_PATH = "%s/pcn.parquet" % PARQUET_PATH
WHITE_PATH = "/mnt/ve_share/songyuhao/generation/data/filtered_p2p_cn/filtered/%s_%s_%s_%s.json" % (MODE, SCENE, PARA, SIZE)
print(WHITE_PATH)
WHITE_FILTER = True

if TYPE == "txt":
    with open (P2P_PATH, "r") as input_file:
        print(P2P_PATH)
        paths = [_.strip().split("/") for _ in input_file.readlines()]
    
elif TYPE == "folder":
    paths = []
    # Iterate over the files in the folder
    for root, dirs, files in tqdm(os.walk(FOLDER_PATH)):
        for file in files:
            # Get the full path of the file
            file_path = os.path.join(root, file)
            paths.append(file_path)

    paths = [_.strip().split("/") for _ in paths]
    print(paths[:10])
    
    for _ in paths:
        _[1] = "share"
        del _[2]
        
    print(paths[:10])
    print(len(paths))

# -5: refine/replce
# -4: scene
# -3: id
# -2: parameter
if WHITE_FILTER:
    with open(WHITE_PATH) as white_file:
        white_json = json.load(white_file)
        white_ids = [_["id"] for _ in white_json]
    ori_lens = len(paths)

    paths = [_ for _ in paths if int(_[-3]) in white_ids]
    print(ori_lens, len(paths))

input_image, edited_image = [], []
for path in tqdm(paths):
    mode = path[-5]
    scene = path[-4]
    id = path[-3]
    parameter = path[-2]
    prompt = path[-1]
    if mode == MODE and scene == SCENE and parameter == PARA:
        if STREET:
            if SCENE == "night":
                if "at_daytime" in prompt or "at_nighttime" in prompt:
                    pass
                else:
                    continue
        else:
            if SCENE == "night":
                if "at_daytime" in prompt or "at_nighttime" in prompt:
                    continue
                else:
                    pass
        
        if scene[:-1] in prompt:
            edited_image.append(path)
        else:
            input_image.append(path)

# print(edited_image[:10])
edited_image = sorted(edited_image, key=lambda x: x[-3])
input_image = sorted(input_image, key=lambda x: x[-3])

print(len(edited_image))
print(len(input_image))

assert len(edited_image) == len(input_image), print(len(edited_image), len(input_image))

sampled_items = random.sample(list(zip(input_image, edited_image)), SIZE)

# Unpack the sampled items into separate lists
input_image, edited_image = zip(*sampled_items)
input_image = ["/".join(_) for _ in input_image]
edited_image = ["/".join(_) for _ in edited_image]

instruction = ["make it %s" % SCENE for _ in range(SIZE)]
assert len(edited_image) == len(input_image) == len(instruction)


print(input_image[:10])
print(edited_image[:10])
print(instruction[:10])


# Create a DataFrame with your data
data = {
    'input_image': input_image,
    'edit_prompt': instruction,
    'edited_image': edited_image,
}

df = pd.DataFrame(data)

# Save the DataFrame as a Parquet file
df.to_parquet(PARQUET_PATH)
print(PARQUET_PATH)
