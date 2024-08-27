import gradio as gr
import numpy as np
import ob1
import cv2

img_size = None
img_heightx = 0
img_widthx = 0
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
    return img_d

description = "This system sorts out beans mixed with stones and groundnuts by giving it a different color. 😊"
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