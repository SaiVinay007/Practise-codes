'''
Make a video from a set of images 
Using opencv
'''
import cv2
import numpy as np
import os
import glob
from os.path import isfile, join

def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    # for sorting the files properly according to the numbers
	# The [3:-4] is because the files of form img12.jpg
    # files.sort() without arguments returns in the lexigraphical order of file names
    files.sort(key = lambda x: int(x[3:-4]))
 
    for i in range(len(files)):
        filename=pathIn + files[i]
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        # inserting the frames into an image array
        frame_array.append(img)
 
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
 
def main():
    pathIn= './video1/'
    pathOut = 'video1.avi'
    fps = 5.0
    convert_frames_to_video(pathIn, pathOut, fps)
 
if __name__=="__main__":
    main()
