import os
import cv2 

output_dir = "/mnt/ve_share/songyuhao/generation/data/p2p_cn_human/0711/"
image_txt = "/mnt/ve_share/songyuhao/generation/data/p2p_cn_human.txt"

with open (image_txt) as input_file:
    paths = [_.strip() for _ in input_file.readlines()]
    
    
    
    
for ind, path in enumerate(paths):
    if ind % 2 == 0:
        image1 = cv2.imread(path)
    else:
        image2 = cv2.imread(path)
        image_combine = cv2.hconcat([image1, image2])
        cv2.imwrite("%s/%d.png" % (output_dir, ind//2), image_combine)              
             
        