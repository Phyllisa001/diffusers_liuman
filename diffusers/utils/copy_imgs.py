import os
from tqdm import tqdm

NAME_DIR = "/mnt/share_disk/syh/data/train/online/diffusions/comb_cls/imgs"
NEW_DIR = "/mnt/share_disk/syh/data/train/online/diffusions/comb_cls_1024/imgs_ori"
OLD_DIR = "/mnt/share_disk/syh/data/train/diffusions/origin/imgs"

name_dic = dict()
for _, _, files in os.walk(NAME_DIR):
    for name in tqdm(files):
        file_name = NAME_DIR + '/' + name
        stem_name = name.split(".")[0]
        coor_name = stem_name.split("-")[-1]
        name_dic[coor_name] = stem_name
        
for short_name in tqdm(list(name_dic.keys())):
    full_name = name_dic[short_name]
    os.system('cp {} {}'.format("%s/%s.jpg" % (OLD_DIR, short_name), "%s/%s.jpg" % (NEW_DIR, short_name)))