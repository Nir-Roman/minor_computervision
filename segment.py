from ultralytics import YOLO
import cv2 as cv
import numpy as np
import torch
import cvzone
import math

class segmentation():
    def __init__(self, capture_index):
       
        self.capture_index = capture_index
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = self.seg_model_load()

    def seg_model_load(self):
        model=YOLO("yolov8n-seg.pt")
        
        return model
    def predict(self, frame):
       
        results = self.model(frame)
        
        return results
    
    def __call__(self):
        print("called")
        cap = cv.VideoCapture(self.capture_index)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            ret,frame=cap.read()

            modell=self.seg_model_load()
            results=self.predict(frame)
            

            for r in results:
                boxes=r.boxes
                for box in boxes:
                    x1,y1,x2,y2= box.xyxy[0]
                    x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)

                    w,h=x2-x1,y2-y1

                    cvzone.cornerRect(frame,(x1,y1,w,h))
                    conf=math.ceil((box.conf[0]*100))/100
                    cvzone.putTextRect(frame,f'{conf}',(max(0,x1),max(35,y1)),scale=3,thickness=1)
                    
            
            
            cv.imshow("YOLOv8_Segmentaion", frame)
 
            if cv.waitKey(5) & 0xFF ==27 :
                
                break

        cap.release()
        cv.destroyAllWindows()

segment = segmentation(capture_index=0)
print("hello")
segment()
            













