import cv2 as cv
import sys
img = cv.imread(cv.samples.findFile("datasets/coco128/images/train2017/000000000009.jpg"))
if img is None:
    sys.exit("Could not read the image.")
cv.imshow("Display window", img)
k = cv.waitKey(0)
if k == ord("s"):
    cv.imwrite("starry_night.png", img)
