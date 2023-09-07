from tqdm import tqdm
import json


scene = "night"
co = "0.80_0.80_2.00"
length = "all"
result_json = "/mnt/ve_share/songyuhao/generation/data/filtered_p2p_cn/ori/replace_blend_reweight_%s_%s_%s.json" % (scene, co, length) if scene != "snowy" else "/mnt/ve_share/songyuhao/generation/data/filtered_p2p_cn/ori/refine_blend_reweight_%s_%s_%s.json" % (scene, co, length)
topk = 500

black_ids = []
with open(result_json) as json_res:
    res = json.load(json_res)
    res_sorted = sorted(res, key=lambda x: x["directional_sim"], reverse=True)

    for i in tqdm(range(len(res_sorted))):
        id_ = res_sorted[i]["id"]
        image_image_sim = res_sorted[i]["image_image_sim"]
        image_caption_sim_1 = res_sorted[i]["image_caption_sim_1"]
        image_caption_sim_2 = res_sorted[i]["image_caption_sim_2"]
        directional_sim = res_sorted[i]["directional_sim"]
        if image_image_sim < 0.75 or image_caption_sim_1 < 0.2 or image_caption_sim_2 < 0.2 or directional_sim < 0.2:
            black_ids.append(id_)
            
white_res_sorted = [_ for _ in res_sorted if _["id"] not in black_ids][:topk]
print("Pass Ratio: ", (1 - round(len(black_ids)/ len(res_sorted), 4)) * 100)
print(white_res_sorted[0])
print(white_res_sorted[-1])

new_result_json = result_json.replace("ori", "filtered").replace("_%s.json" % length, "_%d.json" % min(topk, len(white_res_sorted)))
with open(new_result_json, "w") as json_file:
    json.dump(white_res_sorted, json_file)
print(new_result_json)