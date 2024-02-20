from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
#from sort import *
# Create a new YOLO model from scratch
#model = YOLO('yolov8n.yaml')
cap= cv.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
model = YOLO('best.pt')

classNames=['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']

while(1):
    success,frame=cap.read()
    #gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    framecap=cv.flip(frame,1)
    result=model(framecap,stream=True)
    #detections=np.empty((0,5))
    for r in result:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1
            #print(x1,y1,w,h)
            #cv.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),3)
            cvzone.cornerRect(framecap,(x1,y1,w,h))

            conf=math.ceil((box.conf[0]*100))/100
            print(conf)
            clss=int(box.cls[0])
            cvzone.putTextRect(framecap, f'{classNames[clss]}{conf}', (max(0,x1), max(35,y2)),scale=1, thickness=1)

    cv.imshow('video',framecap)
    
    k=cv.waitKey(1)

    if k==ord("q"):
        break


cv.destroyAllWindows()


            
