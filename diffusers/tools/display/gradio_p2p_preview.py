import gradio as gr
import os
from PIL import Image
import random
import json


root = "/mnt/ve_share/songyuhao/generation/data/p2p_cn/imgs"
json_root = "/mnt/ve_share/songyuhao/generation/data/filtered_p2p_cn"
models = ["image_image_sim", "image_caption_sim_1", "image_caption_sim_2", "directional_sim"]
modes = ["ori", "filtered"]
cos = ["0.80_0.80_2.00", "0.70_0.70_2.00"]
orders = ["DESC", "ASC"]
       
def feedback(feedback_text, model, scene, user):
    print("click")
    output = "%s@%s@%s@%s" % (user, scene, feedback_text, length)
    log_path = "/mnt/ve_share/songyuhao/generation/records/txt/%s.txt" % (model)
    with open(log_path, "a") as input_file:
        input_file.writelines(output + "\n")
        print(output)
   
def get_file_paths(folder_path, ids, co):
    print(folder_path)
    file_paths = []
    for id in ids:
        folder_p = "%s/%s/%s" % (folder_path, id, co)
        for root, dirs, files in os.walk(folder_p):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
                print(file_path)
    return file_paths 

def generation_eval():
    scenes = [f for f in os.listdir("%s/replace_blend_reweight" % root)]
    print(scenes)
    

    def clip_api(model, scene, mode, co, num, order):
        print("Loading: %s %s %s %s %s %s" % (model, scene, mode, co, num, order))
        root_img = "replace_blend_reweight/%s" % scene if scene != "snowy" else "refine_blend_reweight/%s" % scene
        result_json = "%s/%s/%s_%s_all.json" % (json_root, mode, root_img.replace("/", "_"), co) if mode == "ori" else "%s/%s/%s_%s_8000.json" % (json_root, mode, root_img.replace("/", "_"), co)
        
        with open(result_json) as json_res:
            res = json.load(json_res)
        res_sorted = sorted(res, key=lambda x: x[model])[:num] if order == "ASC" else sorted(res, key=lambda x: x[model], reverse=True)[:num]
        selecte_id = [_["id"] for _ in res_sorted]
        ret_imgs = get_file_paths(folder_path="%s/%s" % (root, root_img), ids=selecte_id, co=co)
        # print(ret_imgs)
        global length
        length = len(ret_imgs)
        print("%d Images" % length)
        # ret_imgs = [Image.open(_) for _ in ret_imgs]
        return ret_imgs

    examples = []

    title = "<h1 align='center'>Prompt-to-Prompt 图像展示平台</h1>"

    with gr.Blocks() as demo:
        gr.Markdown(title)
        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.components.Radio(label="分类选择", choices=modes, value=modes[0], elem_id=1)
                co = gr.components.Radio(label="参数选择", choices=cos, value=cos[0], elem_id=2)
                model = gr.components.Radio(label="指标选择", choices=models, value=models[0], elem_id=3)
                scene = gr.components.Radio(label="场景选择", choices=scenes, value=scenes[0], elem_id=4)
                num = gr.components.Slider(minimum=0, maximum=200, step=10, value=60, label="返回图片组数", elem_id=2)
                order = gr.components.Radio(label="顺序选择", choices=orders, value=orders[0], elem_id=5)

                btn = gr.Button("搜索")
            with gr.Column(scale=100):
                out = gr.Gallery(label="检索结果为：").style(grid=4, height=200)

        inputs = [model, scene, mode, co, num, order]
        btn.click(fn=clip_api, inputs=inputs, outputs=[out])
        gr.Examples(examples, inputs=inputs)
    return demo


if __name__ == "__main__":
    with gr.TabbedInterface(
            [generation_eval()],
            ["Prompt-to-Prompt 图像展示"],
    ) as demo:
        demo.launch(
            enable_queue=True,
            share=True,
        )
