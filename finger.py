import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys 

img1 = cv2.imread('./database/101_1.tif',0)  # queryImage
img2 = cv2.imread('./database/103_2.tif',0)  # trainImage

cv2.imshow("Input1",img1)
cv2.waitKey(0)
cv2.imshow("Input2",img2)
cv2.waitKey(0)


def build_filters():
    #returns a list of kernels in several orientations
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 32):
        params = {'ksize':(ksize, ksize), 'sigma':1.0, 'theta':theta, 'lambd':15.0,
                  'gamma':0.02, 'psi':0, 'ktype':cv2.CV_32F}
        kern = cv2.getGaborKernel(**params)
        kern /= 1.5*kern.sum()
        filters.append((kern,params))
    return filters

def process(img, filters):
    # returns the img filtered by the filter list

    accum = np.zeros_like(img)
    for kern,params in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def thin(img):
	size = np.size(img)
	skel = np.zeros(img.shape,np.uint8)
	 
	ret,img = cv2.threshold(img,127,255,0)
	element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	done = False
	 
	while( not done):
	    eroded = cv2.erode(img,element)
	    temp = cv2.dilate(eroded,element)
	    temp = cv2.subtract(img,temp)
	    skel = cv2.bitwise_or(skel,temp)
	    img = eroded.copy()
	 
	    zeros = size - cv2.countNonZero(img)
	    if zeros==size:
	        done = True
	return skel

"""#Gabor Filter
filters = build_filters()
img1 = process(img1, filters)
img2 = process(img2, filters)
cv2.imshow("gabor1",img1)
cv2.waitKey(0)
cv2.imshow("gabor2",img2)
cv2.waitKey(0)
#cv2.destroyAllWindows()"""


#Binarization
_, img1 = cv2.threshold(img1, 8, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
_, img2 = cv2.threshold(img2, 8, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow("Binary1",img1)
cv2.waitKey(0)
cv2.imshow("Binary2",img2)
cv2.waitKey(0)

img1 = cv2.bitwise_not(img1)
img2 = cv2.bitwise_not(img2)

cv2.imshow("Invert1",img1)
cv2.waitKey(0)
cv2.imshow("Invert2",img2)
cv2.waitKey(0)

#Thinning

img1=thin(img1)
cv2.imshow("Thinned1",img1)
cv2.waitKey(0)
img2=thin(img2)
cv2.imshow("Thinned2",img2)
cv2.waitKey(0)


# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)


print("Length:",len(matches))
if(len(matches)<250):
	print("\nAUTHENTICATION ERROR\n")
else:
	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)
	tscore=0

	for i in matches:
		#print(i.distance)
		tscore=tscore+i.distance
	print("total score",tscore)
	if(tscore < (21000)):
		print("yes")
	else:
		print("no")	
	# Draw first 10 matches.
	img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:20],None,flags=2)

	cv2.imshow('Output',img3)
	cv2.waitKey(0)
	cv2.destroyAllWindows()