#made by Frederik Gade and Martin Ægidius, DTU, 2022. 
#Strong inspiration from https://medium.com/@iKhushPatel/convert-video-to-images-images-to-video-using-opencv-python-db27a128a481

import cv2
import os 

path = '/video/'
video = 'SampleVideo_1280x720_10mb.mp4'
path = os.path.join(path,video)

if not os.path.exists(os.path.join(path,'/frames/')):
    os.umask(0)
    os.makedirs(os.path.join(path,'/frames/'),mode=0o777)

vidcap = cv2.VideoCapture(path)
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite("image"+str(count)+".jpg", image)     # save frame as JPG file
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