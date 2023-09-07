import os
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time


def load_and_dowmload_imgs(json_path, save_dir, mode):
    # try:

    
    if mode == "bundle":
        with open(json_path.strip(),'r') as f:
            data_info = json.load(f)
        imgUrl = '/' + data_info['imgUrl']
        save_path = os.path.join(save_dir, os.path.basename(imgUrl))
        os.system('cp {} {}'.format(imgUrl, save_path))
        
    elif mode == "clip":
        with open(json_path.strip(),'r') as f:
            data_info = json.load(f)
        data_os = data_info['camera']
        for data_o in data_os:
            if data_o["name"] == "front_middle_camera":
                imgUrl = '/' + data_o['oss_path']

                # imgUrl = json_path.strip()
                save_path = os.path.join(save_dir, os.path.basename(imgUrl))
                os.system('cp {} {}'.format(imgUrl, save_path))
                break
    elif mode == "whole_clip":
        with open(json_path.strip(),'r') as f:
            data_info = json.load(f)
        time_stamp = json_path.split("_")[-1].split(".")[0]
        date_time = datetime.fromtimestamp(int(str(time_stamp)[:10]))
        timestamp_date = date_time.date()
        timestamp_time = date_time.time()
        if time(11, 23, 10) >= timestamp_time >= time(11, 22, 45):
            save_folder_path = "%s/%s" % (save_dir, time_stamp)
            os.makedirs(save_folder_path, exist_ok=True)
            hardware_path = "/" + data_info["hardware_config_path"]
            hardware_save_path = "%s/hardware_config_path.json" % save_folder_path
            os.system('cp {} {}'.format(hardware_path, hardware_save_path))
            
            lidar_path = "/" + data_info["lidar_merge"][0]["oss_path_pcd_txt"]
            lidar_save_path = "%s/lidar_merge.pcd" % save_folder_path
            os.system('cp {} {}'.format(lidar_path, lidar_save_path))
            
            camera_names = [_["name"] for _ in data_info["camera"]]
            camera_paths = ["/" + _["oss_path"] for _ in data_info["camera"]]
            for i in range(len(camera_paths)):
                camera_name = camera_names[i]
                camera_path = camera_paths[i]
                camera_save_path = "%s/%s.jpg" % (save_folder_path, camera_name)
                os.system('cp {} {}'.format(camera_path, camera_save_path))
                            
            lidar_names = [_["name"] for _ in data_info["lidar"]]
            lidar_paths = ["/" + _["oss_path_pcd_txt"] for _ in data_info["lidar"]]
            for i in range(len(lidar_paths)):
                lidar_name = lidar_names[i]
                lidar_path = lidar_paths[i]
                if lidar_name in ["center_128_lidar_scan_data", "left_M1_lidar_scan_data", "right_M1_lidar_scan_data"]:
                    lidar_save_path = "%s/%s.pcd" % (save_folder_path, lidar_name)
                    os.system('cp {} {}'.format(lidar_path, lidar_save_path))
                    
    elif mode == "pure":
        save_path = os.path.join(save_dir, os.path.basename(json_path))
        os.system('cp {} {}'.format(json_path, save_path))
            
                

jsontxt_path = '/mnt/ve_share/songyuhao/generation/lsu_query/dusk.txt'
# jsontxt_folder = "/share/2d-od/lei/aiday_inf/demo2"
save_dir = '/mnt/share_disk/syh/data/train/diffusions/lsu/imgs/dusk'  
# save_dir = '/mnt/ve_share/yayun'


if not os.path.exists(save_dir):
    os.system('mkdir -p {}'.format(save_dir))



# file_lst, jsons = [], []
# for root, dirs, files in os.walk(jsontxt_folder):
#     for file in files:
#         file_name = os.path.join(root, file)
#         if file_name.endswith(".json"):
#             # print(file_name)
#             jsons.append(file_name)
            
with open(jsontxt_path,'r') as f:
    jsons = f.readlines()


jsons = tqdm(jsons[:1000])

count = 0
mode = "bundle"  # [bundle | clip | whole_clip]
with ThreadPoolExecutor(max_workers=64) as pool:
    temp_infors = [pool.submit(load_and_dowmload_imgs, json_path, save_dir, mode) for json_path in tqdm(jsons)]
    temp_infors = tqdm(temp_infors)
    data_infos = [t.result() for t in temp_infors]
    # for json_path in jsons:
    #     pool.submit(load,json_path,count)
    #     count += 1
    