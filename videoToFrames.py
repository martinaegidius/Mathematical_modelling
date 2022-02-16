#made by Frederik Gade and Martin Aegidius, DTU, 2022. 
#Strong inspiration from https://medium.com/@iKhushPatel/convert-video-to-images-images-to-video-using-opencv-python-db27a128a481

import cv2
import os 

path = os.path.dirname(os.path.abspath(__file__))
vidpath = os.path.join(path,'video/SampleVideo_1280x720_10mb.mp4')
savepath = os.path.join(path,'video/frames')

if not os.path.exists(savepath):
   os.umask(0)
   os.mkdir(savepath,mode=0o777)
   print("new dir made")


vidcap = cv2.VideoCapture(vidpath)
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(os.path.join(savepath,"image"+str(count)+".jpg"), image)     # save frame as JPG file
    return hasFrames

sec = 0
frameRate = 0.5 #//it will capture image in each 0.5 second
count=1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)