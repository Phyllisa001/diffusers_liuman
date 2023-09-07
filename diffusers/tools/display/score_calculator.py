from collections import defaultdict
import os
import pandas as pd

root_path = "/mnt/ve_share/songyuhao/generation/records/txt"
save_path = "/mnt/ve_share/songyuhao/generation/records/xlsx"

selected_user = "syh"

def get_file_paths(folder_path):
    file_paths = []
    for root, directories, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_paths.append(file_path)
    return file_paths

# Call the function to get the file paths
paths = get_file_paths(root_path)

for path in paths:
    correct_dict = defaultdict(int)
    all_dict = defaultdict(int)
    
    with open(path) as result_file:
        ress = [_.strip() for _ in result_file.readlines()]
    
    model = path.split("/")[-1][:-4]
    
    ress = sorted(ress, key=lambda x:x.split("@")[1].split("_")[-1])
    for res in ress:
        res_split = res.split("@")
        user = res_split[0]
        if user == selected_user:
            scene = res_split[1].split("_")[-1].capitalize()
            if model == "INS-Base":
                print(scene)
            if scene not in ["Backlight"]:
                if int(res_split[2]) > int(res_split[3]):
                    continue
                correct_dict[scene] += int(res_split[2])
                all_dict[scene] += int(res_split[3])
    
    scenes = list(correct_dict.keys())
    
    res_dict = dict()
    for scene in scenes:
        res_dict[scene] = round(correct_dict[scene] / all_dict[scene], 4)
    if sum((list(all_dict.values()))) != 0:
        res_dict["Average"] = round(sum(list(correct_dict.values())) / sum((list(all_dict.values()))), 4)
        
    df = pd.DataFrame(res_dict, index=["通过率"])
    print(df)
    print(model)
    print(res_dict)
    output_path = "%s/%s_%s.xlsx" % (save_path, model, selected_user)
    df.to_excel(output_path)
    print(output_path)

    