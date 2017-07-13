import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys 
from main_enhancement import main_enhancement
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
	#ret,img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
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

def compute(i,img):

	"""#Gabor Filter
	filters = build_filters()
	img = process(img, filters)
	cv2.imshow("gabor"+str(i),img)
	cv2.waitKey(0)
	img2 = process(img2, filters)
	cv2.imshow("gabor2",img2)
	cv2.waitKey(0)
	#cv2.destroyAllWindows()"""

	#cv2.imshow("Input"+str(i),img)
	#cv2.waitKey(0)

	img = cv2.equalizeHist(img)
	#cv2.imshow("equalizeHist"+str(i),img)
	#cv2.waitKey(0)

	#Binarization
	_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	#_, img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	#cv2.imshow("Binary"+str(i),img)
	#cv2.waitKey(0)

	#cv2.imshow("Binary2",img2)
	#cv2.waitKey(0)
	path="binary"+str(i)+'.png'
	cv2.imwrite(path,img)
	
	"""img = cv2.bitwise_not(img)
	cv2.imshow("Invert"+str(i),img)
	cv2.waitKey(0)"""

	'''
	img2 = cv2.bitwise_not(img2)
	cv2.imshow("Invert2",img2)
	cv2.waitKey(0)'''

	main_enhancement(path)

	img=cv2.imread(path,-1)
	#Thinning

	img=thin(img)
	#cv2.imshow("Thinned"+str(i),img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	#img2=thin(img2)
	#cv2.imshow("Thinned2",img2)
	#cv2.waitKey(0)
	images.append(img)
	cv2.imwrite("Thinned"+str(i)+".tif",img)
	#cv2.imwrite("Thinned2.png",img2)



	# Initiate ORB detector
	orb = cv2.ORB_create()

	# find the keypoints and descriptors with ORB
	kp, des = orb.detectAndCompute(img,None)
	#kp2, des2 = orb.detectAndCompute(img2,None)

	return kp,des

def match(img1,img2,des1,des2,kp1,kp2):
	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	# Match descriptors.
	matches = bf.match(des1,des2)

	print("Length:",len(matches))
	if(len(matches)<250):
		print("AUTHENTICATION ERROR\n")
	else:
		# Sort them in the order of their distance.
		matches = sorted(matches, key = lambda x:x.distance)
		
		tscore=0
		for i in matches:
			#print(i.distance)
			tscore=tscore+i.distance
		print("\ntotal score:",tscore)
		if(tscore < (20000)):
			print("AUTHENTICATION SUCCESSFUL\n")
		else:
			print("AUTHENTICATION FAILURE\n")
		# Draw first 10 matches.
		img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:20],None,flags=2)

		cv2.imshow('Output',img3)
		cv2.waitKey(0)

descriptors=[]
keypoints=[]
images=[]
while(1):
	print("\n\tFINGERPRINT")
	print("1. ADD")
	print("2. CHECK")
	print("3. EXIT")
	ch=input("\nEnter your choice:")
	if(ch=="1"):
		for i in range(1,9):
			#img = cv2.imread('./database/101_'+str(i)+'.tif',0)
			path='./database/012_3_'+str(i)+'.tif'
			img = cv2.imread(path,0)
			x,y=compute(i,img)
			keypoints.append(x)
			descriptors.append(y)
	elif(ch=="2"):
		#path=input("\nEnter path (./database/101_1.tif):")
		path=input("\nEnter path ( ./database/012_3_1.tif ):")
		img = cv2.imread(path,0)
		x,y=compute(0,img)
		for i in range(len(descriptors)):
			match(images[i],images[len(images)-1],descriptors[i],y,keypoints[i],x)
		cv2.destroyAllWindows()
	elif(ch=="3"):
		exit(0)
	else:
		print("\nInvalid choice\n")

#img1 = cv2.imread('./database/101_1.tif',0)  # queryImage
#img2 = cv2.imread('./database/101_4.tif',0)  # trainImage

'''cv2.imshow("Input1",img1)
cv2.waitKey(0)
cv2.imshow("Input2",img2)
cv2.waitKey(0)'''


#equ = cv2.equalizeHist(img)

