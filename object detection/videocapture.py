import cv2 as cv
import sys

cap= cv.VideoCapture(0)

while(1):
    ret,frame=cap.read()
    gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.imshow('video',gray)

    k=cv.waitKey(1)

    if k==ord("q"):
        break

cap.release() 
cv.destroyAllWindows()