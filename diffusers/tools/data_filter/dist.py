import json
import matplotlib.pyplot as plt
import numpy as np  
import os

scene = "night"
co = "0.80_0.80_2.00"
length = "1000"
mode = "filtered"  # ["filtered" | " ori"]
root = "/mnt/ve_share/songyuhao/generation/data/filtered_p2p_cn"

# result_json = "%s/%s/replace_blend_reweight_%s_%s_%s.json" % (root, mode, scene, co, length) if scene != "snowy" else "%s/ori/refine_blend_reweight_%s_%s_%s.json" % (root, mode, scene, co, length)
# folder_name = "%s/%s_%s_%s" % (root, scene, mode, length)

result_json = "/mnt/ve_share/songyuhao/generation/data/filtered_p2p_cn/filtered/night_0.80+0.70_1000.json"
folder_name = "/mnt/ve_share/songyuhao/generation/data/filtered_p2p_cn/night_0.80+0.70_1000"

os.makedirs(folder_name, exist_ok=True)

with open(result_json) as json_res:
    res = json.load(json_res)

keys = [_ for _ in res[0].keys() if _ != "id"]

for key in keys:
    values = [_[key] for _ in res]    

    mean = np.mean(values)
    std_dev = np.std(values)
    medain = np.median(values)

    plt.figure(figsize=(15, 10))
    plt.hist(values, bins=100, color='skyblue', edgecolor='black')

    plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
    plt.axvline(mean + std_dev, color='orange', linestyle='dashed', linewidth=2, label=f'Standard Deviation: {std_dev:.2f}')
    plt.axvline(mean - std_dev, color='orange', linestyle='dashed', linewidth=2)
    plt.axvline(medain, color='green', linestyle='dashed', linewidth=2, label=f'Median: {medain:.2f}')

    plt.legend()
    
    name = "%s_%s_%s_%s" % (scene, key, mode, length)

    # Adding labels and title
    plt.xlabel('Value', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title(name, fontsize=16)

    # Save the plot as a PNG image
    plt.savefig('%s/%s.png' % (folder_name, key))
    plt.clf()

print(folder_name)
