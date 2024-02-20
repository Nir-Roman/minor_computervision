from ultralytics import YOLO
import numpy as np
import cv2 as cv
import torch
import cvzone

class ObjectTracking:
    def __init__(self, capture_index):
       
        self.capture_index = capture_index
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = self.load_model()

    def load_model(self):
       
        model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n model
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

                    cvzone.cornerRect(frame,(x1,y1,w,h))
                    print(type(track))
            
            
            cv.imshow("YOLOv8_Detection", frame)
 
            if cv.waitKey(5) & 0xFF == :
                
                break

        cap.release()
        cv.destroyAllWindows()

tracker = ObjectTracking(capture_index=0)
print("hello")
tracker()
            




