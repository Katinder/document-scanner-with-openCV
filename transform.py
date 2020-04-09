import numpy as np
import cv2

def order_points(pts):
	rect=np.zeros((4,2),dtype="float32")
	s=pts.sum(axis=1)
	rect[0]=pts[np.argmin(s)]#top left has the smallest sum
	rect[2]=pts[np.argmax(s)]#bottom right will have the largest sum
	diff=np.diff(pts,axis=1)
	rect[1]=pts[np.argmin(diff)] #top right has the smallest diff (considering x-y, i.e. signs too)
	rect[3]=pts[np.argmax(diff)] #bottom left will have the largest diff

	return rect

def four_point_transform(image,pts):

	rect=order_points(pts)
	(tl,tr,br,bl)=rect

	widthA=np.sqrt(((br[0]-bl[0])**2)+((br[1]-bl[1])**2))
	widthB=np.sqrt(((tr[0]-tl[0])**2)+((tr[1]-tl[1])**2))
	maxWidth=max(int(widthA),int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	#set of destination points for birds eye view
	dst=np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]],dtype="float32")

	M=cv2.getPerspectiveTransform(rect,dst)
	warped=cv2.warpPerspective(image,M,(maxWidth,maxHeight))

	return warped
