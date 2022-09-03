import cv2
import numpy as np

thres = 0.5

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

className =[]
classpath = "label.txt"
with open(classpath, "rt") as f:
    className = f.read().rstrip("\n").split("\n")

configpath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weigthpath = "frozen_inference_graph.pb"
#create model
model = cv2.dnn_DetectionModel(weigthpath,configpath)
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

if __name__ == "__main__":

    while True:
        success, img = cap.read()
        (classIds, confs, bbox) = model.detect(img, confThreshold=thres)
        print(classIds,confs,bbox)

        if len(classIds) != 0 and np.all(np.array(classIds) < 81):
            for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img,box,color=(0,255,0))

                cv2.putText(img,className[classId-1],(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                cv2.putText(img, str(np.round(conf*100)), (box[2], box[3]), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 0, 255), 2)

        cv2.imshow("output", img)
        cv2.waitKey(1)