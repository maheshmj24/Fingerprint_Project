# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:42:58 2016

@author: utkarsh
"""

import numpy as np
import matplotlib.pylab as plt;
import scipy.ndimage
import sys

import enhance_image

def main_enhancement(path):
    img = scipy.ndimage.imread(path);
        
    if(len(img.shape)>2):
        # img = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
        img = np.dot(img[...,:3], [0.299, 0.587, 0.114]);

    rows,cols = np.shape(img);
    aspect_ratio = np.double(rows)/np.double(cols);

    new_rows = 350;             # randomly selected number
    new_cols = new_rows/aspect_ratio;

    #img = cv2.resize(img,(new_rows,new_cols));
    img = scipy.misc.imresize(img,(np.int(new_rows),np.int(new_cols)));

    enhanced_img = enhance_image.image_enhance(img);    

    #if(1):
    print('saving the image')
    #scipy.misc.imsave("enhanced",enhanced_img);
    scipy.misc.imsave(path,enhanced_img);
    #else:
    #    plt.imshow(enhanced_img,cmap = 'Greys_r');
