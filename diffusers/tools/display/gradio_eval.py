import gradio as gr
import os
from PIL import Image

root = "/mnt/ve_share/songyuhao/generation/data/result/diffusions/vis/instructpix2pix/official"
users = ["syh", "fkx", "lyy", "ckl", "others"]

def feedback(feedback_text, model, scene, user):
    print("click")
    output = "%s@%s@%s@%s" % (user, scene, feedback_text, length)
    log_path = "/mnt/ve_share/songyuhao/generation/records/txt/%s.txt" % (model)
    with open(log_path, "a") as input_file:
        input_file.writelines(output + "\n")
        print(output)
   
def get_file_paths(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths 

def generation_eval():
    # print("request.headers: ", request.headers)
    models = sorted([f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))])
    model_root = "%s/%s" % (root, models[0])
    scenes = [f for f in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, f))]
    display_scenes = [_.split("_")[-1] for _ in scenes]

    def clip_api(model, scene):
        print("Loading: %s %s" % (model, scene))
        # global feedback_btn
        # feedback_btn = gr.Radio.update(choices=['未知(反馈可持续优化LSU🤗)', '准确率低(<30%)', '准确率中(30% ~ 80%)', '准确率高(>80%)',], value='未知(反馈可持续优化LSU🤗)')
        ret_imgs = get_file_paths("%s/%s/make_it_%s" % (root, model, scene))
        # print(ret_imgs)
        global length
        length = len(ret_imgs)
        print("%d Images" % length)
        ret_imgs = [Image.open(_) for _ in ret_imgs]
        return ret_imgs

    examples = []
    # examples = [
    #     ["道路上工作的环卫工人", 20, VLP, "是"],
    #     ["雨天打伞的人", 20, VLP, "是"],
    #     ["救护车", 20, VLP, "是"],
    #     ["校车", 20, VLP, "是"],
    #     ["警车", 20, VLP, "是"],
    #     ["喝水的人", 20, VLP, "是"],
    #     ["后备箱打开", 20, VLP, "是"],
    # ]

    title = "<h1 align='center'>图像生成人工评测平台</h1>"

    with gr.Blocks() as demo:
        gr.Markdown(title)
        # gr.Markdown(description)
        with gr.Row():
            with gr.Column(scale=1):
                # with gr.Column(scale=2):
                #     text = gr.Textbox(value="道路上工作的环卫工人", label="请填写文本", elem_id=0, interactive=True)

                # num = gr.components.Slider(minimum=0, maximum=200, step=1, value=30, label="返回图片数", elem_id=2)
                model = gr.components.Radio(label="模型选择", choices=models, value=models[0], elem_id=3)
                scene = gr.components.Radio(label="场景选择", choices=display_scenes, value=display_scenes[0], elem_id=4)
                # thumbnail = gr.components.Radio(label="是否返回缩略图", choices=["yes", "no"], value="yes", elem_id=4)
                btn = gr.Button("搜索")
                # feedback_btn = gr.Radio(label="检索结果反馈",
                #       choices=['未知(反馈可持续优化LSU🤗)', '准确率低(<30%)', '准确率中(30% ~ 80%)', '准确率高(>80%)',], value='未知(反馈可持续优化LSU🤗)')
            with gr.Column(scale=100):
                out = gr.Gallery(label="检索结果为：").style(grid=6, height=200)
                feedback_btn = gr.Textbox(value="", label="请填写合格图像数", elem_id=0, interactive=True)
                user = gr.components.Radio(label="用户选择", choices=users, value=users[0], elem_id=2)
                btn2 = gr.Button("提交")
            
        # inputs = [text, num, model, thumbnail]
        inputs = [model, scene]
        btn.click(fn=clip_api, inputs=inputs, outputs=[out])
        btn2.click(feedback, inputs=[feedback_btn, model, scene, user])
        # feedback_btn.change(feedback, inputs=feedback_btn)
        gr.Examples(examples, inputs=inputs)
    return demo


if __name__ == "__main__":
    with gr.TabbedInterface(
            [generation_eval()],
            ["图像生成人工评测"],
    ) as demo:
        demo.launch(
            enable_queue=True,
            share=True,
        )
