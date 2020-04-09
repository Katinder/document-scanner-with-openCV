from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local #for black and white image
import numpy as np
import argparse
import cv2
import imutils #functions for resizing,rotating and cropping images
import random as rng

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to the image to be scanned")
args=vars(ap.parse_args())

#step 1: edge detection
image=cv2.imread(args["image"])
ratio=image.shape[0]/500.0
orig=image.copy()
image=imutils.resize(image,height=500)

#convert to grayscale,blur it and then find edges
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray=cv2.GaussianBlur(gray,(5,5),0) #blur to remove high frequency noise
edged=cv2.Canny(gray,75,200)

#show original and edged detected images
print("edge detecion")
cv2.imshow("image: ",image)
cv2.imshow("edged: ",edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

#step 2: finding contours
cnts=cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnts=imutils.grab_contours(cnts)
cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:5]

# canny_output=edged.copy()
# contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     # Draw contours
# drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

# for i in range(len(contours)):
# 	color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
# 	cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
#     # Show in a window
# cv2.imshow('Contours', drawing)

for c in cnts:
	peri=cv2.arcLength(c,True)
	approx=cv2.approxPolyDP(c,0.02*peri,True)

	if len(approx)==3:
		screenCnt=approx
		break

# if (not screenCnt):
# 	scrennCnt=cnts[0];

print("Find contours of paper: ")
cv2.drawContours(image,[screenCnt],-1,(0,255,0),2)
cv2.imshow("Outline: ",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# step 3: perspective transform and threshold
warped=four_point_transform(orig,screenCnt.reshape(4,2)*ratio)

warped=cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)#convert to grayscale
T=threshold_local(warped,11,offset=10,method="gaussian") #threshold it to black and white
warped=(warped>T).astype("uint8")*255

print("apply perspective transform")
cv2.imshow("orginal: ",imutils.resize(orig,height=650))
cv2.imshow("scanned: ",imutils.resize(warped,height=650))
cv2.waitKey(0)