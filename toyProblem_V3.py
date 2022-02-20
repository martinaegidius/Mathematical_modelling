import os
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib
import numpy as np
from scipy import ndimage as ndi


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
       self.LUKASBOI(N=2,stride=10)
       #self.quiver_plot()
       self.simple_plot()
        
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
        
    def LUKASBOI(self,N=7,stride=40):
        ''' Does Lukas Kanabe on all images in variable self.grayImages.

        Args:
            N (int): N is one side in the square around the pixels to be processed. (Defualt is 3)
            stride (int): The stride for pixels to process. (Default is 40)

        Returns:
            Nothing. Does however add variables to class instance. The primary are:
            self.coordinates, an M long list containing sublists with coordinates to
            pixels being processed. 
            self.gradiants, an k x M long list of lists containing matching gradiants
            to coordinates self.coordinates for every frame.      
        '''
        if N < 1:
            N=3
        if (stride < N or stride < 1):
            stride = 40
            if stride<N:
                stride=N+5
        
        minX = N
        minY = N
        maxX = len(self.grayImages[0,:,0])-N
        maxY = len(self.grayImages[0,0,:])-N

        N = int(N/2)
        
        # global coordinates
        # global gradiants
        
        
        coordinates = []
        for i in range(minX,maxX,stride):
            for j in range (minY,maxY,stride):
                coordinates.append([i,j])

        self.coordinates = coordinates
        self.gradiants = np.empty((len(self.grayImages),len(self.coordinates),2,1))

        for imageNumber, image in enumerate(self.grayImages):    
            for i in range(minX,maxX,stride):
                for j in range(minY,maxY,stride):
                    gradX = ndi.gaussian_filter(self.grayImages[:,i-N:i+N+1,j-N:j+N+1],sigma=N,order=(0,1,0),mode='nearest')
                    gradY = ndi.gaussian_filter(self.grayImages[:,i-N:i+N+1,j-N:j+N+1],sigma=N,order=(0,0,1),mode='nearest')
                    vT = ndi.gaussian_filter(self.grayImages[:,i-N:i+N+1,j-N:j+N+1],sigma=N,order=(1,0,0),mode='nearest')



                    A = np.concatenate((np.reshape(gradX[imageNumber], (-1,1)),np.reshape(gradY[imageNumber],(-1,1))),axis=1)
                    b = np.reshape(vT[imageNumber],(-1,1))
                    
                    if j == minY:
                        self.testOutput = np.linalg.lstsq(A,b,rcond=None)[0:1]
                    else:
                        self.testOutput = np.concatenate((self.testOutput,np.linalg.lstsq(A,b,rcond=None)[0:1]))

                if (i==minX):
                    self.mellemstop = self.testOutput
                else:
                    self.mellemstop = np.concatenate((self.mellemstop,self.testOutput))

            self.gradiants[imageNumber]=self.mellemstop



    def simple_plot(self):
        # global xygrads
        # global p

        xygrads = self.gradiants.squeeze() #remove unecessary 4th dim
        p = self.coordinates
        
        #one image frame
        '''
        fig = plt.imshow(self.grayImages[10,:,:],cmap='gray',vmin=0,vmax=1)
        plt.plot(p[7][0],p[7][1],'o',color='red',markersize=2)
        plt.plot(np.array([p[7][0],p[7][0]+2*xygrads[10,7,0]]),np.array([p[7][1],p[7][1]+2*xygrads[10,7,1]]),'-',color='red',markersize=15)
        plt.xlim([p[7][0]-25,p[7][0]+25])
        plt.ylim([p[7][1]-25,p[7][1]+25])
        plt.show()
        '''
        
        
        
        for i in range(len(self.grayImages)-1): #imagewise  #previous was 43
            fig = plt.imshow(self.grayImages[i,:,:],cmap='gray') #plot frame of interest
            for j in range(0,xygrads.shape[1]-1):
                
                if (abs(xygrads[i,j,1])+abs(xygrads[i,j,0])<5) or (abs(xygrads[i,j,1])+abs(xygrads[i,j,0])>20):
                    pass
                else:
                    plt.plot(p[j][1],p[j][0],'o',color='red',markersize=1)
                    plt.plot(np.array([p[j][1],p[j][1]-2*xygrads[i,j,1]]),np.array([p[j][0],p[j][0]-2*xygrads[i,j,0]]),'-',color='red',markersize=15)
            plt.xlim([0,256])
            plt.ylim([256,0])
            plt.show()
            
        # #bemærk punkt (53,103). det er tilsvarende p[7] -> xygrads[:,7,:]. Der bør være stor bevægelse = stor gradient
        # pointOfInterest = np.array(xygrads[:,7,:])
        # mean = np.mean(pointOfInterest)
        # print(str(mean) + "den er meget lille")
        # #sammenlign med punkt [143,43] = p[93], hvor der bør være nærmest ingen bevægelse
        # pointOfInterest = np.array(xygrads[:,93,:])
        # mean = np.mean(pointOfInterest)
        # print(str(mean) + "den er ret stor")
        
        
                
        


    def quiver_plot(self):
        '''function which takes as input an array which defines the 
        coordinates for pixels for which flow-lines are requested 
        in format (x,y). 
        the pixel-array is to be matched with corresponding values stored in self.testOutput in some sort of sensible way.
        '''
        
        #generating a diagonal pointsample (should come from other function)
        #p = np.linspace(0,256,num = 30).astype(int) #pairwise array of pixels requested analyzed (should come from LUKASBOI)
        global q 
        #q = np.vstack((p,p)).T
        q = np.array(self.coordinates)
        
        global dataHolder ##create a matrix for holding all gradients for points. Should be generated from another function
        #in format [frames x pointvalues (pairwise)]
        #dataHolder = np.zeros((len(self.grayImages),len(q)))
        
        dataHolder = self.gradiants
        print(dataHolder.shape)
        #concatenate all flow measurements
        #for j in range(len(self.grayImages)-1):
            #for i in range(0,len(q),2):
                #we are loosing last frame. Should maybe be corrected to len(grayImages)
                
                #following should be a oneliner somehow 
       #         dataHolder[j][i] = self.testOutput[j][0]
       #         dataHolder[j][i+1] = self.testOutput[j][1]
                #thus this is not completely dynamic, sicing would fix
        
                
        
        ##step for making proper normalized colorbar
        #part 1 remove outliers from dataset
        global lengths 
        lengths = np.zeros((dataHolder.shape[0],dataHolder.shape[1])) #for storing vector norms for each pixel for each frame
        for j in range(dataHolder.shape[0]): #loop over frames 
            for i in range(0,dataHolder.shape[1]): #loop over pixel of interest
                lengths[j][i] = np.linalg.norm(dataHolder[j,i,:,:]) ##fill every col of lengths
        
        #gross outlier-removal (floor outliers to mean value)
        #tolerance = 100*np.mean(lengths)
        #lengths[lengths>tolerance] = np.mean(lengths) #remember that norms are always positive, thus no abs() needed
        upperLimit = (np.percentile(lengths,95)) #define upper edge of colorbar as 95 percentile of lengths
        #upperLimit = np.amax(lengths)
        lowerLimit = np.percentile(lengths,70)
        lengths[lengths<lowerLimit] = np.NAN
        
        #create colorbar element which forces normalization to range of lengths
        norm = matplotlib.colors.Normalize(vmin=lowerLimit,vmax=upperLimit,clip=False)
        norm.autoscale(lengths)
        
       #plot all images with colorbar. 
        numImages = 50#dataHolder.shape[0] #last frame has been lost earlier :-(
        
        for i in range(numImages): #imagewise 
            fig = plt.imshow(self.grayImages[i,:,:],cmap='gray') #plot frame of interest
            for j in range(0,dataHolder.shape[1]): #loop over all columns in dataHolder-array for the corresponding frame
                #print(j) for debugging
                #plt.quiver(q[j,0], q[j,1], dataHolder[i,j,0,:], dataHolder[i,j,1,:], lengths[i,j],cmap='jet',norm=norm)#arguments: startpoint(x,y),vector(x,y),"measurement" for colormap, create arrow-colors
                plt.plot(q[j,0],q[j,1],'o',color='red',markersize=2)
                plt.plot(np.array([q[j,0],q[j,1]+2*dataHolder[i,j,1,:]]),np.array([q[j,1],q[j,0]+2*dataHolder[i,j,0,:]]),'-',color = 'red', markersize = 15)
                
                
            #plt.colorbar()
            #plt.clim(lowerLimit,upperLimit) #max should be max of array after outliers rmvd
            plt.xlim([0,250]) #forces constant axes
            plt.ylim([250,0]) #forces constant axes (reversed) 
            plt.show()
        #test if last frame looks proper
        print("last bottom right arrow should have color corresponding to " +str(lengths[numImages-1,-1]))
        print("last bottom right arrow should have direction x: "+str(dataHolder[numImages-1,-1,0,:])+" and y: "+ str(dataHolder[numImages-1,-1,1,:]))#whoops - only for debugging. 

OpticalFlow()
