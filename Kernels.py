#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 08:26:41 2022

@author: max
"""
from optical_flow import * 
import scipy.ndimage 
import numpy as np
import matplotlib.pyplot as plt

#kernelArr = np.zeros((3,3))
filterHolder = np.empty((4,256,256))
filterHolder[0,:,:] = scipy.ndimage.prewitt(images[0])
filterHolder[1,:,:] = scipy.ndimage.sobel(images[0])
filterHolder[2,:,:] = scipy.ndimage.prewitt(images[0],axis=0)
filterHolder[3,:,:] = scipy.ndimage.sobel(images[0],axis=0)

titles = ["Prewitt columnwise","Sobel columnwise","Prewitt rowwise","Sobel rowwise"]

#generating subplots
rows = 2
cols = 2
axes = []
fig = plt.figure()
for a in range(rows*cols):
    axes.append(fig.add_subplot(rows,cols,a+1))
    #subplot_title=("Subplot" + str(a))
    axes[-1].set_title(titles[a])
    plt.imshow(filterHolder[a,:,:])
fig.tight_layout()
plt.show()


#seperate image-filtering: 
filterGradX = np.empty((len(images),256,256))
filterGradY = np.empty((len(images),256,256))
filterGradT = np.empty((len(images)))

for i in range(len(images)):
    filterGradX[i,:,:] = scipy.ndimage.prewitt(images[i,:,:],axis=-1)#row-orientation filtering
    filterGradY[i,:,:] = scipy.ndimage.prewitt(images[i,:,:],axis=0)#column-orientation filtering
    #time filter? what are we supposed to do? 
    
#x gradient comparison
def gradient_compare(dim,x_grad,y_grad,filterGradX,filterGradY):
    if dim.casefold()=="y".casefold():
        for i in range(len(images)):
            #create subplot for comparisons
            rows = 1
            cols = 2
            axes = []
            fig = plt.figure()
            for a in range(rows*cols):
                axes.append(fig.add_subplot(rows,cols,a+1))
                #subplot_title=("Subplot" + str(a))
                if a%2==0:
                    axes[-1].set_title("Gradient")
                    plt.imshow(x_grad[i,:,:])
                else: 
                    axes[-1].set_title("Prewitt filter")
                    plt.imshow(filterGradX[i,:,:])
            fig.tight_layout()
            plt.show()
            
    if dim.casefold()=="x".casefold():
        for i in range(len(images)):
            #create subplot for comparisons
            rows = 1
            cols = 2
            axes = []
            fig = plt.figure()
            for a in range(rows*cols):
                axes.append(fig.add_subplot(rows,cols,a+1))
                #subplot_title=("Subplot" + str(a))
                if a%2==0:
                    axes[-1].set_title("Gradient")
                    plt.imshow(y_grad[i,:,:])
                else: 
                    axes[-1].set_title("Prewitt filter")
                    plt.imshow(filterGradY[i,:,:])
            fig.tight_layout()
            plt.show()

#y gradient comparison


gradient_compare("x",x_grad,y_grad,filterGradX,filterGradY)
#for i in range(10):
#    plt.imshow(y_grad[i,:,:])
    #plt.pause(0.3)
#    plt.show()
    
    


