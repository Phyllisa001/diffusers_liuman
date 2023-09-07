import os
import pickle
from tqdm import tqdm
from collections import Counter, defaultdict

DICT_DIR = "/mnt/ve_share/songyuhao/generation/data/train/diffusions/comb_ins/dict"
PMPS_DIR = "/mnt/ve_share/songyuhao/generation/data/train/diffusions/comb_ins/pmps"
RES_DIR = "/mnt/ve_share/songyuhao/generation/data/train/diffusions/comb_ins/pmps_seg_some"

os.makedirs(RES_DIR, exist_ok=True)

mask_dict = {
    0:"Bird",
    1:"Ground_Animal",
    2:"Curb",
    3:"Fence",
    4:"Guard_Rail",
    5:"Barrier",
    6:"Wall",
    7:"Bike_Lane",
    8:"Crosswalk_Plain",
    9:"Curb_Cut",
    10:"Parking",
    11:"Pedestrian_Area",
    12:"Rail_Track",
    13:"Road",
    14:"Service_Lane",
    15:"Sidewalk",
    16:"Bridge",
    17:"Building",
    18:"Tunnel",
    19:"Person",
    20:"Bicyclist",
    21:"Motorcyclist",
    22:"Other_Rider",
    23:"Lane_Marking_Crosswalk",
    24:"Lane_Marking_General",
    25:"Mountain",
    26:"Sand",
    27:"Sky",
    28:"Snow",
    29:"Terrain",
    30:"Vegetation",
    31:"Water",
    32:"Banner",
    33:"Bench",
    34:"Bike_Rack",
    35:"Billboard",
    36:"Catch_Basin",
    37:"CCTV_Camera",
    38:"Fire_Hydrant",
    39:"Junction_Box",
    40:"Mailbox",
    41:"Manhole",
    42:"Phone_Booth",
    43:"Pothole",
    44:"Street_Light",
    45:"Pole",
    46:"Traffic_Sign_Frame",
    47:"Utility_Pole",
    48:"Traffic_Light",
    49:"Traffic_Sign_Back",
    50:"Traffic_Sign_Front",
    51:"Trash_Can",
    52:"Bicycle",
    53:"Boat",
    54:"Bus",
    55:"Car",
    56:"Caravan",
    57:"Motorcycle",
    58:"On_Rails",
    59:"Other_Vehicle",
    60:"Trailer",
    61:"Truck",
    62:"Wheeled_Slow",
    63:"Car_Mount",
    64:"Ego_Vehicle"
}

my_mask_dict_1 = {
    2:"Curb",
    3:"Fence",
    4:"Guard Rail",
    5:"Barrier",
    6:"Wall",
    7:"Bike Lane",
    8:"Crosswalk",
    9:"Curb Cut",
    10:"Parking",
    11:"Pedestrian Area",
    12:"Rail Track",
    13:"Road",
    14:"Service_Lane",
    15:"Sidewalk",
    16:"Bridge",
    17:"Building",
    18:"Tunnel",
    19:"Person",
    20:"Bicyclist",
    21:"Motorcyclist",
    23:"Crosswalk",
    24:"Lane Marking",
    25:"Mountain",
    26:"Sand",
    27:"Sky",
    28:"Snow",
    29:"Terrain",
    30:"Vegetation",
    31:"Water",
    44:"Street_Light",
    45:"Pole",
    46:"Traffic Sign Frame",
    47:"Utility Pole",
    48:"Traffic Light",
    49:"Traffic Sign Back",
    50:"Traffic Sign Front",
    52:"Bicycle",
    54:"Bus",
    55:"Car",
    56:"Caravan",
    57:"Motorcycle",
    60:"Trailer",
    61:"Truck",
}

number_filter = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,23,24,25,26,27,28,29,30,31]
filter_lst = list(my_mask_dict_1.keys())
seg_dic = dict()

# # ===============================================================================
# for _, _, files in os.walk(DICT_DIR):
#     for name in tqdm(files):
#         dict_name = DICT_DIR + '/' + name
#         stem_name = name.split(".")[0]
#         with open(dict_name, 'rb') as f:
#             dic = pickle.load(f)
#             filter_dic = {k: v for k, v in dic.items() if v in filter_lst}
#             item_dic = {k: my_mask_dict_1[v] for k, v in filter_dic.items()}
#             item_counter = (Counter(item_dic.values()))
#             # with number
#             # item_des = ["%d %s" % (v, k) for k, v in item_counter.items()]
#             # all without number
#             item_des = ["%s" % k for k, v in item_counter.items()]

#             seg_dic[stem_name] = ", ".join(item_des)
# # ===============================================================================

# ===============================================================================
# some number
for _, _, files in os.walk(DICT_DIR):
    for name in tqdm(files):
        dict_name = DICT_DIR + '/' + name
        stem_name = name.split(".")[0]
        with open(dict_name, 'rb') as f:
            dic = pickle.load(f)
            dic_1 = {k: v for k, v in dic.items() if v in number_filter}
            dic_2 = {k: v for k, v in dic.items() if v not in number_filter}
            
            filter_dic_1 = {k: v for k, v in dic_1.items() if v in filter_lst}
            filter_dic_2 = {k: v for k, v in dic_2.items() if v in filter_lst}
            
            item_dic_1 = {k: my_mask_dict_1[v] for k, v in filter_dic_1.items()}
            item_dic_2 = {k: my_mask_dict_1[v] for k, v in filter_dic_2.items()}
            
            item_counter_1 = (Counter(item_dic_1.values()))
            item_counter_2 = (Counter(item_dic_2.values()))
            
            # # with number
            item_des_2 = ["%d %s" % (v, k) for k, v in item_counter_2.items()]
            # # all without number
            item_des_1 = ["%s" % k for k, v in item_counter_1.items()]

            seg_dic[stem_name] = ", ".join(item_des_2 + item_des_1)
# ===============================================================================
            

pmp_dic = dict()
for _, _, files in os.walk(PMPS_DIR):
    for name in tqdm(files):
        file_name = PMPS_DIR + '/' + name
        stem_name = name.split(".")[0]
        with open(file_name, 'r') as f:
            blip_desc = f.readline()
            pmp_dic[stem_name] = blip_desc


merge_dic = defaultdict(list)
for d in (pmp_dic, seg_dic): # you can list as many input dicts as you want here
    for key, value in d.items():
        merge_dic[key].append(value)
        
merge_dic = {k: ", ".join(v) for k, v in merge_dic.items()}

for k, v in tqdm(merge_dic.items()):
    save_path = "%s/%s.txt" % (RES_DIR, k)
    with open(save_path, "w") as output_file:
        output_file.writelines(v)
        