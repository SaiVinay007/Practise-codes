#%%
import cv2
import numpy as np
from matplotlib import pyplot as plt

##########################################################
#%%

cap = cv2.VideoCapture('/home/saivinay/Documents/imgpro/slow.flv')

#reading first frame
ret,frame = cap.read()

r,h,c,w = 127,50,193,75
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h,c:c+w]
hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((15.,135.,135.)), np.array((360.,255.,255.)))
print (mask.max())
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
        cv2.imshow("image", dst)
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        # print(x,y,w,h)
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
# cv2.destroyAllWindows()
# cap.release()
# print (frame[r:r+h,c:c+w,:])
# print(frame.shape)
# print(frame[250:340,:])
print(dst)
print(dst.shape)
plt.imshow(dst)
plt.show

















############################################################
#%%
cap = cv2.VideoCapture('/home/saivinay/Documents/imgpro/vid1.mov')

ret , frame = cap.read()
cv2.imwrite('vimg.jpg',frame)


##########################################################
#%%

cap = cv2.VideoCapture('/home/saivinay/Documents/imgpro/vid1.mov')

#reading first frame
ret,frame = cap.read()

r,h,c,w = 161,140,277,40
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
        
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        # print(x,y,w,h)
        cv2.imshow('img2',img2)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        # else:
        #     cv2.imwrite(chr(k)+".jpg",img2) 
    else:
        break
cv2.destroyAllWindows()
cap.release()