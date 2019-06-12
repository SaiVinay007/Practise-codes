import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Inputs to the code")
parser.add_argument("--video_path",type=str,default='/home/saivinay/Documents/ai/ComputerVision/Implementations/opencv/vid1.mov',help="path to video file")
args = parser.parse_args()



### optical flow
def optical_flow(cap):

    # params for ShiTomasi corner detection
    features_params = dict(maxCorners = 100,qualityLevel = 0.5,minDistance = 1,blockSize = 7)

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors27xq
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    ret,oldframe = cap.read()
    # print (oldframe)
    old_gray = cv2.cvtColor(oldframe,cv2.COLOR_RGB2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray,mask=None,**features_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(oldframe)

    while(1):
        ret,frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1,st,err = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,p0,None,**lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask,(a,b),(c,d),color[i].tolist(),2)
            frame = cv2.circle(frame,(a,b),2,color[i].tolist(),-1)
        img = cv2.add(frame,mask)

        cv2.imshow('frame',img)
        if cv2.waitKey(100) & 0xff == ord('q'):
            break

        # Now update the previous frame and previous points    
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

    cv2.destroyAllWindows()
    cap.release()



### Dense optical flow

def dense_optical_flow(cap):
    
    ret, frame1 = cap.read()
    
    prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
    
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    
    while(1):
        ret, frame2 = cap.read()
        next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
        cv.imshow('frame2',bgr)
        if cv.waitKey(10) & 0xFF==ord('q'):
        	break
        prvs = next

    cap.release()
    cv.destroyAllWindows()

if __name__=='__main__':
    
    cap = cv2.VideoCapture(args.video_path)

    optical_flow(cap)
    dense_optical_flow(cap)