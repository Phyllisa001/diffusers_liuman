import json

SCENE = "night"
MODE = "replace_blend_reweight" if SCENE != "snowy" else "refine_blend_reweight"
PARA = "0.80_0.80_2.00"
SIZE = 8000
FOLDER_PATH = "/mnt/ve_share/songyuhao/generation/data/p2p_cn/imgs/%s/%s" % (MODE, SCENE)
path = "/mnt/ve_share/songyuhao/generation/data/filtered_p2p_cn/filtered/%s_%s_0.80_0.80_2.00_1000.json" % (MODE, SCENE, PARA, SIZE)
topk = 50

with open (path, "r") as input_file:
    json_file = json.load(input_file)
    

print(json_file)