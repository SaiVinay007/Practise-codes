#%%
import cv2
import numpy as np
from matplotlib import pyplot as plt


#%%

cap = cv2.VideoCapture('/home/saivinay/Documents/imgpro/slow.flv')

#reading first fraqme
ret,frame = cap.read()

r,h,c,w = 127,50,193,75
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h,c:c+w]
hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv_roi, np.array((135.,135.,135.)), np.array((360.,255.,255.)))

#
roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        
        ret , track_window = cv2.CamShift(dst,track_window,term_crit)

        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame,[pts],True,255,2)
        cv2.imshow('img2',img2)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
        # else:
        #     cv2.imwrite(chr(k)+".jpg",img2) 
    else:
        break
cv2.destroyAllWindows()
cap.release()

#%%
# np.set_printoptions(threshold=np.nan)
print(dst)
