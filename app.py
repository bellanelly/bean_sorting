import gradio as gr
from PIL import Image
# import torch
import matplotlib.pyplot as plt
import io
import numpy as np
# from ob1 import *
import ob1
import cv2



def model_inference():
    pass

# COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
#           [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

img_size = None

img_heightx = 0
img_widthx = 0

# def process_class_list(classes_string: str):
#     return [x.strip() for x in classes_string.split(",")] if classes_string else []

def model_inference(img_x, model_name: str, prob_threshold: int, classes_to_show = str):
    global img_heightx , img_widthx, img_size

    model=cv2.dnn.readNetFromONNX(model_name)

    cvimg = cv2.cvtColor(np.array(img_x), cv2.COLOR_RGB2BGR)


    img_size = cvimg.shape[:2]


    img_height, img_width = img_size

    img_heightx, img_widthx =   int(img_height/2), int(img_width/2)
    cvimg = cv2.resize(cvimg, (img_widthx, img_heightx))




    myclasses = ob1.class_names()

    pil_img = True

    
    width,height=640,640

    img_d = ob1.detections(model, cvimg, width, height, myclasses, pil_img)

    return img_d #plot_results(img_d)


description = """
Yolov5 beans, groundnuts, stones detection 😊
"""

# image_in = gr.components.Image()

# if img_size:
#     image_out = gr.components.Image(container=False, width=img_widthx, height=img_heightx)

# image_out = gr.components.Image()
# model_choice = gr.components.Dropdown(["best.onnx"], value="best.onnx", label="YOLOV5")
# prob_threshold_slider = gr.components.Slider(minimum=0, maximum=1.0, step=0.01, value=0.9, label="Probability Threshold")
# classes_to_show = gr.components.Textbox(placeholder="e.g. person, car , laptop", label="Classes to use (Optional)")

# Iface = gr.Interface(
#     fn=model_inference,
#     inputs=[image_in,model_choice, prob_threshold_slider, classes_to_show],
#     outputs=image_out,
#     title="Object Detection With YOLOv5",
#     description=description,
#     theme="NoCrypt/miku",
#     #gradio.themes.colors.gray.
#     #theme='HaleyCH/HaleyCH_Theme',
# ).launch()


# import gradio as gr
# import numpy as np
# import ob1
# import cv2

# def model_inference(img_x, model_name: str, prob_threshold: float, classes_to_show: str):
#     model = cv2.dnn.readNetFromONNX(model_name)
#     cvimg = cv2.cvtColor(np.array(img_x), cv2.COLOR_RGB2BGR)
    
#     img_height, img_width = cvimg.shape[:2]
#     img_heightx, img_widthx = int(img_height/2), int(img_width/2)
#     cvimg = cv2.resize(cvimg, (img_widthx, img_heightx))
    
#     myclasses = ob1.class_names()
#     pil_img = True
#     width, height = 640, 640
    
#     img_d = ob1.detections(model, cvimg, width, height, myclasses, pil_img)
#     return img_d

description = "Yolov5 beans, groundnuts, stones detection 😊"

with gr.Blocks(theme="NoCrypt/miku") as demo:
    gr.Markdown("# Object Detection With YOLOv5")
    gr.Markdown(description)
    
    with gr.Column():
        image_in = gr.Image(label="Input Image")
        model_choice = gr.Dropdown(["best.onnx"], value="best.onnx", label="YOLOV5")
        prob_threshold_slider = gr.Slider(minimum=0, maximum=1.0, step=0.01, value=0.9, label="Probability Threshold")
        classes_to_show = gr.Textbox(placeholder="e.g. person, car , laptop", label="Classes to use (Optional)")
        detect_button = gr.Button("Detect Objects")
        image_out = gr.Image(label="Output Image")
    
    detect_button.click(
        fn=model_inference,
        inputs=[image_in, model_choice, prob_threshold_slider, classes_to_show],
        outputs=image_out
    )

demo.launch()