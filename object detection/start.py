from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
from sort import *
# Create a new YOLO model from scratch
#model = YOLO('yolov8n.yaml')
cap= cv.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
model = YOLO('yolov8n.pt')

className=[]
#tracking
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)



while(1):
    success,frame=cap.read()
    #gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    result=model(frame,stream=True)
    detections=np.empty((0,5))
    for r in result:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1
            #print(x1,y1,w,h)
            #cv.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),3)
            cvzone.cornerRect(frame,(x1,y1,w,h))

            conf=math.ceil((box.conf[0]*100))/100
            print(conf)
            cvzone.putTextRect(frame, f'{conf}', (max(0,x1), max(35,y2)))
            #class 
            
            clss=int(box.cls[0])

            currentClass=className[clss]



            if currentClass=="car" or currentClass=="motorbike" or currentClass=='truck' or currentClass=="bus" and conf >0.3:
                cvzone.putTextRect(frame, f'{currentClass}{conf}', (max(0,x1), max(35,y2)))
                cvzone.cornerRect(frame,(x1,y1,w,h),l=9)
                currentarray=np.arrary([x1,y1,x2,y2,conf])
                detections=np.vstack(detections,currentarray)


    resultsTracker=tracker.update(detections)
    for result in resultsTracker:
        x1,y1,x2,y2,id=result
        x1,y1,x2,y2=x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)

        w,h=x2-x1,y2-y1
        cvzone.cornerRect(frame,(x1,y1,w,h),l=9)
        cvzone.putTextRect(frame, f'{currentClass}{conf}', (max(0,x1), max(35,y2)))









    cv.imshow('video',frame)

    k=cv.waitKey(1)

    if k==ord("q"):
        break

cap.release() 
cv.destroyAllWindows()

# Load a pretrained YOLO model (recommended for training)
#model = YOLO('yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
#results = model.train(data='coco128.yaml', epochs=3)

# Evaluate the model's performance on the validation set
##results = model.val()

# Perform object detection on an image using the model
#results = model('/home/rockingstarniraj/Desktop/object detection/datasets/coco128/images/train2017/000000000009.jpg', show=True)

# Export the model to ONNX format
#success = model.export(format='onnx')
#cv2.waitKey(0)