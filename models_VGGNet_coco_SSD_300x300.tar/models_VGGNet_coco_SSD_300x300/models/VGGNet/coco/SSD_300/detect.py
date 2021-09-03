import time
import numpy as np
import cv2
from cv2 import dnn

#make label and color list
labels = ["background", "aeroplane", "bicycle", "bird", "boat", 
          "bottle", "bus", "car", "cat", "chair", 
          "cow", "diningtable", "dog", "horse", "motorbike", 
          "person", "pottedplant", "sheep",  "sofa", "train", "tvmonitor"]
colors = np.random.uniform(0, 255, size=(len(labels), 3))

#prepare test image
image = cv2.imread('./data/1510/')
(h, w) = image.shape[:2]
blob = dnn.blobFromImage(image, 1, (512, 512))

#prepare model network
prototxt = "./src_512/deploy.prototxt"
model = "./src_512/VGG_coco_SSD_300x300_iter_400000.caffemodel"
net = dnn.readNetFromCaffe(prototxt, model)

#feed in image and get result
net.setInput(blob)
t = time.time()
prob = net.forward()
print("Runtime:", time.time()-t)

#diaplay result
for i in np.arange(0, prob.shape[2]):
    confidence = prob[0, 0, i, 2]
    if confidence > 0.4:    #change threshold value to get the result you want 
        # get data from prob
        index = int(prob[0, 0, i, 1])
        box = prob[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x, y, endX, endY) = box.astype("int")
        color = colors[index]
        # draw rect
        cv2.rectangle(image, (x, y), (endX, endY), color, 2)
        # draw label
        label = "{}: {:.2f}%".format(labels[index], confidence * 100)
        print("{}".format(label))
        (fontX, fontY) = cv2.getTextSize(label, cv2.FONT_HERSHEY_TRIPLEX, 0.5, 1)[0]
        y = y + fontY if y-fontY<0 else y
        cv2.rectangle(image,(x, y-fontY),(x+fontX, y),color,cv2.FILLED)
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0), 1)

cv2.imwrite("./results/test.jpg", image)
