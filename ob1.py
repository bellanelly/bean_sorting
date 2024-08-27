import cv2
import numpy as np

def class_names():
    classes=[]
    with open("classes.txt","r") as file:
        for line in file.readlines():
            myclass=line.strip("\n")
            classes.append(myclass)

        return classes
    

# my_class_names=class_names()


def detections(model,image,model_width,model_height, my_class_names, pil_image):
    blob = cv2.dnn.blobFromImage(image, 1/255 , (model_width,model_height), swapRB=True, mean=(0,0,0), crop= False)
    model.setInput(blob)
    
    outputs= model.forward(model.getUnconnectedOutLayersNames())
    
    out= outputs[0]

    n_detections= out.shape[1]
    height,width=image.shape[:2]
    x_scale= width/model_width
    y_scale= height/model_height

    conf_threshold= 0.3
    score_threshold= 0.5
    nms_threshold=0.5



    class_ids=[] 
    score=[] 
    boxes=[]

    

    for i in range(n_detections):
                detect=out[0][i]
                confidence= detect[4]
                if confidence >= conf_threshold:
                    class_score= detect[5:]
                    class_id= np.argmax(class_score)
                    if (class_score[class_id]> score_threshold):
                        score.append(confidence)
                        class_ids.append(class_id)
                        x, y, w, h = detect[0], detect[1], detect[2], detect[3]
                        left= int((x - w/2)* x_scale )
                        top= int((y - h/2)*y_scale)
                        width = int(w * x_scale)
                        height = int(h *y_scale)
                        box = np.array([left, top, width, height])
                        boxes.append(box)

    indices = cv2.dnn.NMSBoxes(boxes, np.array(score), conf_threshold, nms_threshold)
    #print(indices)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        if "bean" in my_class_names[class_ids[i]]:
            print(my_class_names[class_ids[i]])
            cv2.rectangle(image, (left, top), (left + width, top + height), (0, 255, 0), 2)
        else:
            cv2.rectangle(image, (left, top), (left + width, top + height), (255, 0, 0), 2)
        label = "{}:{:.2f}".format(my_class_names[class_ids[i]], score[i])
            
        print(label)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 1)
        dim, baseline = text_size[0], text_size[1]
        img = cv2.rectangle(image, (left, top), (left + dim[0], top + dim[1] + baseline), (0,0,0), cv2.FILLED)
        img = cv2.putText(image, label, (left, top + dim[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1, cv2.LINE_AA)
        print('predictions: ', box, class_ids[i], score[i] )

    if pil_image:
         img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

    return img



if __name__=="__main__":

    model=cv2.dnn.readNetFromONNX( "best.onnx")

    my_class_names=class_names()

    image=cv2.imread(r"c:\Users\EliteBook\Desktop\b3\3 (5).jpg")

    # image= cv2.resize(image,(500,700))

    pil_img = False
    
    width,height=640,640

    img_det = detections(model,image,width,height, my_class_names, pil_img)

    cv2.imwrite("n6.jpg",img_det)

    cv2.imshow("detected image",img_det)
    cv2.waitKey(0)