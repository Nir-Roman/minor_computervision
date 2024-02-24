from ultralytics import YOLO
import numpy as np
import cv2 as cv
import torch
import cvzone
import math

class ObjectTracking:
    def __init__(self, capture_index):
       
        self.capture_index = capture_index
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = self.load_model()

    def load_model(self):
       
        model = YOLO("modeln.pt")  # load a pretrained YOLOv8n model
        #model.fuse()
    
        return model
    
    def predict(self, frame):
       
        results = self.model(frame)
        
        return results
    
    def tracker(self,frame):
        resultt=self.model.track(frame, persist=True)
        return resultt
    
    def __call__(self):
        print("called")
        cap = cv.VideoCapture(self.capture_index)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

        classes_name= ['Bike', 'Buildings', 'Car', 'Human', 'NoParking', 'Parking', 'Pole', 'Road', 'Vehicle', 'dustbin', 'grass', 'small-obstacles', 'trees']

        while True:
            ret,frame=cap.read()

            modell=self.load_model()
            results=self.predict(frame)
            track=self.tracker(frame)

            for r in results:
                boxes=r.boxes
                for box in boxes:
                    x1,y1,x2,y2= box.xyxy[0]
                    x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)

                    w,h=x2-x1,y2-y1

                    conf=math.ceil((box.conf[0]*100))/100
                    clss=int(box.cls[0])
                    cvzone.putTextRect(frame,f'{classes_name[clss]}{conf}',(max(0,x1),max(35,y1)),scale=3,thickness=1)
                    
                    #print(type(track))

            
            
            cv.imshow("YOLOv8_Detection", frame)
 
            if cv.waitKey(5) & 0xFF == 27:
                
                break

        cap.release()
        cv.destroyAllWindows()

tracker = ObjectTracking(capture_index=0)
print("hello")
tracker()
            




