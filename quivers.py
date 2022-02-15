#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 09:50:22 2022

@author: max
"""

#moule for quivers 

import numpy as np 
from matplotlib import pyplot as plt 
import os 
import matplotlib.image as img

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

path = os.path.dirname(__file__)
path = os.path.join(path,'../../toyProblem_F22/')
#path = '/Users/frederikgade/Documents/DTU/6. Semester/02526 Mathematical Modeling/Excersize 1/toyProblem_F22/'
imageList = os.listdir(path)
images = [img.imread(path+os.sep+imageName) for imageName in sorted(imageList)]
arrayList = [np.asarray(image) for image in images]
grayImages = np.array([rgb2gray(colorImage) for colorImage in arrayList])


p = np.array([50,50])
vector = np.array([100,-100])
fig = plt.imshow(grayImages[0,:,:],cmap='gray')
plt.quiver(p[0], p[1], vector[0], vector[1],color=['red'])
plt.xlim([0,250]) #forces constant axes
plt.ylim([250,0]) #forces constant axes
plt.show()

 #m, clim=[-2,2])
for i in range(10):
    plt.imshow(grayImages[i,:,:],cmap='gray',vmin=0,vmax=1)
    plt.plot(p[0],p[1],'o',color='red',markersize=2)
    plt.plot(np.array([p[1],p[1]+2*vector[1]]),np.array([p[0],p[0]+2*vector[0]]),'-',color='red',markersize=15)
    #plt.xlim([p[1]-25,p[1]+25])
    #plt.ylim([p[0]-25,p[0]+25])
    
    #vector = np.array([50,50])
    #origin = np.array([50,50])
    #plt.quiver(origin[0],origin[1],vector[0],vector[1],color=['red'],scale=0.2,angles='xy')
    
    
    
    #origin = np.array([[0, 12, 12], [0, 0, 0]])
    #plt.quiver(*origin, data[:, 0], data[:, 1], color=['black'], scale=15)

    plt.show()
#plt.rcParams["figure.figsize"] = [7.00, 3.50]
#plt.rcParams["figure.autolayout"] = True

#plt.show()


