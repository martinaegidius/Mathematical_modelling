import os
# import matplotlib as mpl
# mpl.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib
import numpy as np
from scipy import ndimage as ndi
import PIL 
import math
#from numpy import linalg as LA

class OpticalFlow():
    def __init__(self):
        self.runAll()
        
    def runAll(self):
       self.prepImages()
       self.gradientCalc()
    #    self.showGradies(self.xGrad)
       self.kernelConv()
    #    self.showGradies(self.sobelKernelX)
       self.gaussConvelution()
       #self.showGradies(self.gaussConvTest)
       #self.CornerPixelGradient()
       self.LUKASBOI()
       self.quiver_plot()
        
    def prepImages(self):
        path = os.path.dirname(__file__)
        path = os.path.join(path,'../../toyProblem_F22/')
        #path = '/Users/frederikgade/Documents/DTU/6. Semester/02526 Mathematical Modeling/Excersize 1/toyProblem_F22/'
        imageList = os.listdir(path)
        images = [img.imread(path+os.sep+imageName) for imageName in sorted(imageList)]
        arrayList = [np.asarray(image) for image in images]
        self.grayImages = np.array([self.rgb2gray(colorImage) for colorImage in arrayList])
        
    def rgb2gray(self,rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def gradientCalc(self):
        self.xGrad = np.empty((len(self.grayImages[:,0,0]),256,255))
        self.yGrad = np.empty((len(self.grayImages[:,0,0]),255,256))
        for i, dim in enumerate(self.grayImages[:,0,0]):
           for j, dim in enumerate(self.grayImages[0,:,0]):
               self.xGrad[i,j,:] = self.grayImages[i,j,1:]-self.grayImages[i,j,0:-1]
               self.yGrad[i,:,j] = self.grayImages[0,1:,j]-self.grayImages[0,0:-1,j]
   
    def showGradies(self,arr):
        plt.gray()
        for i,boi in enumerate(arr[:,0,0]):
            plt.imshow(arr[i,:,:])
            plt.pause(0.02)
            plt.show

    def kernelConv(self):
        self.sobelKernelX = np.empty((len(self.grayImages[:,0,0]),256,256))
        self.sobelKernelY = np.empty((len(self.grayImages[:,0,0]),256,256))
        for i in range(len(self.grayImages)):
            self.sobelKernelX[i,:,:] = ndi.sobel(self.grayImages[i,:,:],axis=1)
            self.sobelKernelY[i,:,:] = ndi.sobel(self.grayImages[i,:,:],axis=0)

    def gaussConvelution(self,sigmaChoice=4):
        self.gaussConv = np.empty((len(self.grayImages[:,0,0]),256,256))
        self.tVec = np.linspace(0,63,num=64)
        for i in range(len(self.grayImages)):
            for j in range(len(self.grayImages[0][:,0])):
                self.gaussConv[i,:,j] = ndi.gaussian_filter1d(self.grayImages[i,:,j],sigma=sigmaChoice)
        
        for i in range(len(self.grayImages)):
            for j in range(len(self.grayImages[0][0,:])):
                self.gaussConv[i,j,:] = ndi.gaussian_filter1d(self.grayImages[i,j,:],sigma=sigmaChoice)

        self.tVec = ndi.gaussian_filter1d(self.tVec,sigma=sigmaChoice)


        self.gaussConvTest = np.empty((len(self.grayImages[:,0,0]),256,256))
        self.gaussConvTest = ndi.gaussian_filter(self.grayImages,sigma=sigmaChoice,order=(0,0,1))


    def CornerPixelGradient(self):
        sigmaChoice=4
        pixel = [12,12]
        #finding cornerpoint for loop -> [11,11]
        pixelGradX = np.empty((3,3))
        pixelGradY = np.empty((3,3))
        pixel = [11,11]
        
        self.analPix = self.grayImages[0,11:14,11:14]
        for j in range(0,3):
            pixelGradX[:,j] = ndi.gaussian_filter1d(self.analPix[:,j],sigma=sigmaChoice)
            pixelGradY[j,:] = ndi.gaussian_filter1d(self.analPix[j,:],sigma=sigmaChoice)
        
    def LUKASBOI(self,N=3,stride=0):
        N = math.floor(N/2)
        for i in range(len(self.grayImages)-1):
            image = self.grayImages[i]
            image2 = self.grayImages[i+1]
            
            gradY = ndi.gaussian_filter(image[10-N:10+N+1,10-N:10+N+1],sigma=2,order=(0,1))
            gradX = ndi.gaussian_filter(image[10-N:10+N+1,10-N:10+N+1],sigma=2,order=(1,0))
            A = np.concatenate((np.reshape(gradX, (-1,1)),np.reshape(gradY,(-1,1))),axis=1)
    
            vT = -(image2[10-N:10+N+1,10-N:10+N+1]-image[10-N:10+N+1,10-N:10+N+1])
            vT = np.reshape(vT,(-1,1))
            
            if i == 0:
                testOutput = np.linalg.lstsq(A,vT,rcond=None)[0:1]
            else:
                testOutput = np.concatenate((testOutput,np.linalg.lstsq(A,vT,rcond=None)[0:1]))
            
            self.testOutput = testOutput
        
    def quiver_plot(self,p= np.array([10,10])):
        ### function which takes as input an array which defines the 
        ### coordinates for pixels for which flow-lines are requested 
        ### in format (x,y). 
        ### the pixel-array is to be matched with corresponding values stored in self.testOutput in some sort of sensible way.
        
        p = np.array([10,10]) #pairwise array of pixels requested analyzed
        
        
        ##step for making proper normalized colorbar
        #part 1 remove outliers
        
        completeData = np.zeros((63,len(p))) #create storematrix for x y vals
        for i in range(len(p)):
            completeData[:,i] = np.linalg.norm(self.testOutput[i][:])
        
        lengths = np.zeros((63))
        for i in range(63):
            lengths[i] = np.linalg.norm(self.testOutput[i][:])
            #print(lengths[i])
        print("mean is " + str(np.mean(lengths)))
        #gross outlier-removal (floor outliers to mean value)
        tolerance = 3*np.mean(lengths)
        for i in range(len(lengths)):
            if(lengths[i]>=tolerance):
                #print("entered if")
                lengths[i] = np.mean(lengths)
        upperLimit = (max(lengths))
        
        #create colorbar element which forces normalization to range of lengths
        #norm = matplotlib.colors.Normalize()
        norm = matplotlib.colors.Normalize(vmin=0,vmax=upperLimit)
        norm.autoscale(lengths)
        #cm = matplotlib.cm.hsv
        #sm = matplotlib.cm.ScalarMappable(cmap=cm,norm=norm)
        #sm.set_array([])
       
       #plot all images with colorbar. 
        numImages = 32
        
        for i in range(numImages): #imagewise 
            fig = plt.imshow(self.grayImages[i,:,:],cmap='gray')
            plt.quiver(p[0], p[1], self.testOutput[i][0], self.testOutput[i][1], lengths[i],cmap='hsv',norm=norm)#np.linalg.norm(self.testOutput[i][:]),cmap='hsv',norm=norm) #arguments: startpoint(x,y),vector(x,y),"measurement" for colormap, create arrow-colors
            plt.colorbar()
           #plt.clim(0,upperLimit) #max should be max of array after outliers rmvd
            plt.xlim([0,250]) #forces constant axes
            plt.ylim([250,0]) #forces constant axes (reversed) 
            plt.show()
        #test if last frame looks proper
        print("last arrow should have color corresponding to " +str(lengths[i]))
        print("last arrow should have direction x: "+str(self.testOutput[numImages][0])+" and y: "+str(self.testOutput[numImages][1]))
        print("Something seems to be off with arrow-colors.")


OpticalFlow()


























# # plt.gray()
# # for image in grayImages:
# #     plt.imshow(image)
# #     plt.pause(0.02)
# #     plt.show

# # for imageName in imageList:
# #     image = img.imread(path+os.sep+imageName)
# #     print(image.dtype)
# #     print(image.shape)
# #     plt.imshow(image)
# #     plt.show()


