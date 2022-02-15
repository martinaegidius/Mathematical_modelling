#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 11:05:50 2022

@author: max
"""

import numpy as np 
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import os 
from glob import glob
from skimage.color import rgb2gray



src_dir = '/home/max/Desktop/toyProblem_F22/'
#image loading 
image_list = sorted(glob(os.path.join(src_dir,"*.png")))
images = np.empty((len(image_list),256,256))
for i in range(len(image_list)):
    images[i] = np.array(rgb2gray(mpimg.imread(image_list[i]))) #overvej at definer egen rgb2gray

#image plotting as animation 
plt.gray()
for i in range(len(image_list)):
    plt.imshow(images[i,:,:])
    #plt.pause(0.0009)
    plt.show()


#low-level gradient calculation - problem 2.1: 
timeVector = np.linspace(0,63,num = 64) #create arbitrary time-vector. Same as image-index in images

x_grad = np.empty((len(image_list),256,255))
y_grad = np.empty((len(image_list),255,256))
t_grad = np.empty((len(image_list),256,256))

for j in range(len(timeVector)):
    for i in range(256):
        x_grad[j,i,:] = images[j,i,1:] - images[j,i,0:-1]
        y_grad[j,:,i] = images[j,1:,i] - images[j,0:-1,i]

        
#time-gradient 
t_grad = timeVector[1:] - timeVector[0:-1]

for i in range(10):
    plt.imshow(y_grad[i,:,:])
    #plt.pause(0.3)
    plt.show()
#for at checke om du får det rigtige ud: 
#      - Tjek gennemsnittet af gradienter i y vs x. I de første få frames med "hop" bør y være større end x.      
