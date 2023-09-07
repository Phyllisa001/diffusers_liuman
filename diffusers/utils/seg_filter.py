import os
import pickle
from tqdm import tqdm
from collections import Counter, defaultdict
from multiprocessing.pool import ThreadPool


DICT_DIR = "/mnt/share_disk/syh/data/train/diffusions/10w_512/dict"
PMPS_DIR = "/mnt/share_disk/syh/data/train/diffusions/10w_512/pmps"
IMAGE_DIR = "/mnt/share_disk/syh/data/train/diffusions/10w_512/imgs_blip"
NEW_DIR_INS = "/mnt/share_disk/syh/data/train/diffusions/selected/ins_20k"
NEW_DIR_CLS = "/mnt/share_disk/syh/data/train/diffusions/selected/cls_20k"

# DICT_DIR = "/mnt/ve_share/songyuhao/generation/data/train/diffusions/5000/dict"
# PMPS_DIR = "/mnt/ve_share/songyuhao/generation/data/train/diffusions/5000/pmps"
# IMAGE_DIR = "/mnt/ve_share/songyuhao/generation/data/train/diffusions/5000/imgs"
# RES_DIR = "/mnt/ve_share/songyuhao/generation/data/train/diffusions/5000/pmps_seg_test555"
# NEW_DIR_INS = "/mnt/ve_share/songyuhao/generation/data/train/diffusions/5000/new"



# os.makedirs(RES_DIR, exist_ok=True)

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
seg_dic_instance = dict()
seg_dic_class = dict()


name_dic = dict()
for _, _, files in os.walk(PMPS_DIR):
    for name in tqdm(files):
        file_name = PMPS_DIR + '/' + name
        stem_name = name.split(".")[0]
        coor_name = stem_name.split("-")[-1]
        name_dic[coor_name] = stem_name

DICT_DIR = "/mnt/ve_share/songyuhao/generation/data/train/diffusions/lsu/lsu_combine/dict"
NEW_DIR = "/mnt/ve_share/songyuhao/generation/data/train/diffusions/lsu"

        
for short_name in list(name_dic.keys()):
    full_name = name_dic[short_name]
    # os.system('cp {} {}'.format("%s/%s.png" % (IMAGE_DIR, full_name), "%s/imgs/%s.png" % (NEW_DIR_CLS, full_name)))
    # os.system('cp {} {}'.format("%s/%s.txt" % (PMPS_DIR, full_name), "%s/pmps/%s.txt" % (NEW_DIR_CLS, full_name)))
    os.system('cp {} {}'.format("%s/%s.pkl" % (DICT_DIR, short_name), "%s/dict/%s.pkl" % (NEW_DIR_CLS, full_name)))
        
# ===============================================================================
for _, _, files in os.walk(DICT_DIR):
    for name in tqdm(files):
        dict_name = DICT_DIR + '/' + name
        stem_name = name.split(".")[0]
        with open(dict_name, 'rb') as f:
            dic = pickle.load(f)
            filter_dic = {k: v for k, v in dic.items() if v in filter_lst}
            item_dic = {k: my_mask_dict_1[v] for k, v in filter_dic.items()}
            item_counter = (Counter(item_dic.values()))
            
            instance_number = sum(list(item_counter.values()))
            class_number = len(list(item_counter.values()))

            seg_dic_instance[stem_name] = instance_number
            seg_dic_class[stem_name] = class_number
            
            
sorted_seg_dic_instance = dict(sorted(seg_dic_instance.items(), key=lambda item: item[1]))
sorted_seg_dic_class = dict(sorted(seg_dic_class.items(), key=lambda item: item[1]))

for short_name in list(sorted_seg_dic_instance.keys())[:10000]:
    full_name = name_dic[short_name]
    # os.system('cp {} {}'.format("%s/%s.png" % (IMAGE_DIR, full_name), "%s/imgs/%s.png" % (NEW_DIR_INS, full_name)))
    # os.system('cp {} {}'.format("%s/%s.txt" % (PMPS_DIR, full_name), "%s/pmps/%s.txt" % (NEW_DIR_INS, full_name)))
    os.system('cp {} {}'.format("%s/%s.pkl" % (DICT_DIR, short_name), "%s/dict/%s.pkl" % (NEW_DIR_INS, full_name)))
    
for short_name in list(sorted_seg_dic_instance.keys())[-10000:]:
    full_name = name_dic[short_name]
    # os.system('cp {} {}'.format("%s/%s.png" % (IMAGE_DIR, full_name), "%s/imgs/%s.png" % (NEW_DIR_INS, full_name)))
    # os.system('cp {} {}'.format("%s/%s.txt" % (PMPS_DIR, full_name), "%s/pmps/%s.txt" % (NEW_DIR_INS, full_name)))
    os.system('cp {} {}'.format("%s/%s.pkl" % (DICT_DIR, short_name), "%s/dict/%s.pkl" % (NEW_DIR_INS, full_name)))
    
for short_name in list(sorted_seg_dic_class.keys())[:10000]:
    full_name = name_dic[short_name]
    # os.system('cp {} {}'.format("%s/%s.png" % (IMAGE_DIR, full_name), "%s/imgs/%s.png" % (NEW_DIR_CLS, full_name)))
    # os.system('cp {} {}'.format("%s/%s.txt" % (PMPS_DIR, full_name), "%s/pmps/%s.txt" % (NEW_DIR_CLS, full_name)))
    os.system('cp {} {}'.format("%s/%s.pkl" % (DICT_DIR, short_name), "%s/dict/%s.pkl" % (NEW_DIR_CLS, full_name)))
    
for short_name in list(sorted_seg_dic_class.keys())[-10000:]:
    full_name = name_dic[short_name]
    # os.system('cp {} {}'.format("%s/%s.png" % (IMAGE_DIR, full_name), "%s/imgs/%s.png" % (NEW_DIR_CLS, full_name)))
    # os.system('cp {} {}'.format("%s/%s.txt" % (PMPS_DIR, full_name), "%s/pmps/%s.txt" % (NEW_DIR_CLS, full_name)))
    os.system('cp {} {}'.format("%s/%s.pkl" % (DICT_DIR, short_name), "%s/dict/%s.pkl" % (NEW_DIR_CLS, full_name)))
    

# merge_dic = defaultdict(list)
# for d in (pmp_dic, seg_dic): # you can list as many input dicts as you want here
#     for key, value in d.items():
#         merge_dic[key].append(value)
        
# merge_dic = {k: ", ".join(v) for k, v in merge_dic.items()}

# for k, v in tqdm(merge_dic.items()):
#     save_path = "%s/%s.txt" % (RES_DIR, k)
#     with open(save_path, "w") as output_file:
#         output_file.writelines(v)
        