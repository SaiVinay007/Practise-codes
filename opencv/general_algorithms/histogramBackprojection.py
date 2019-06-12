#%%
import cv2
import numpy as np
from matplotlib import pyplot as plt


#%%
target = cv2.imread('/home/saivinay/Documents/imgpro/img2.jpg')
hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)

src_pts = np.array([[30,280],[100,280],[30,320],[280,320]])
dst_pts = np.array([[0,0],[70,0],[0,40],[70,40]])

h, status = cv2.findHomography(src_pts,dst_pts)
object = cv2.warpPerspective(image,h,(70,40))
hsv = cv2.cvtColor(object,cv2.COLOR_BGR2HSV)

plt.imshow(object)
plt.show()

#%%

# cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
histogram = cv2.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])
# cv2.normalize(original_image, normalized_image, arr, alpha=0.0(lower), beta=1.0(upper), norm_type=cv2.NORM_MINMAX)
cv2.normalize(histogram,histogram,0,255,cv2.NORM_MINMAX)

#cv2.calcBackProject(images, channels, hist, ranges, scale[, dst])
dst = cv2.calcBackProject([hsvt],[0,1],histogram,[0,180,0,256],1)

print(dst.shape)
print(target.shape)

# dst = cv2.merge((dst,dst,dst))
# res = cv2.bitwise_and(target,dst)

# Now convolute with circular disc
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
cv2.filter2D(dst,-1,disc,dst)
# threshold and binary AND
ret,thresh = cv2.threshold(dst,50,255,0)
thresh = cv2.merge((thresh,thresh,thresh))
res = cv2.bitwise_and(target,thresh)


plt.imshow(res)
plt.show()


#%%
