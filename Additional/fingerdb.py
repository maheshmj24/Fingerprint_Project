import cv2
import numpy as np
orb = cv2.ORB_create()

# Read Images
train = cv2.imread('101_1.tif',0)
test = cv2.imread('101_3.tif',0)

# Find Descriptors    
kp1,trainDes1 = orb.detectAndCompute(train, None)
kp2,testDes2  = orb.detectAndCompute(test, None)

# Create BFMatcher and add cluster of training images. One for now.
bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False) # crossCheck not supported by BFMatcher
clusters = np.array([trainDes1])
bf.add(clusters)

# Train: Does nothing for BruteForceMatcher though.
bf.train()

matches = bf.match(testDes2)
matches = sorted(matches, key = lambda x:x.distance)

# Since, we have index of only one training image, 
# all matches will have imgIdx set to 0.
print('Length:',len(matches))
for i in range(len(matches)):
    print(matches[i].imgIdx)