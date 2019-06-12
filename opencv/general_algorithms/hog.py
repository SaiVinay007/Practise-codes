#%%
import cv2
import numpy as np
from matplotlib import pyplot as plt

#%%
im_src = cv2.imread('/home/saivinay/Documents/imgpro/img4.jpg')
# cv2.imshow('img',img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

pts_src = np.array([[484,112],[663,112],[484,600],[663,600]])
pts_dst = np.array([[0,0],[64,0],[0,128],[64,128]])

h, status = cv2.findHomography(pts_src, pts_dst)
im_out = cv2.warpPerspective(im_src, h, (64,128))
     
RGB_im_src = cv2.cvtColor(im_src,cv2.COLOR_BGR2RGB)
RGB_im_out = cv2.cvtColor(im_out,cv2.COLOR_BGR2RGB)

# plt.figure()
# plt.subplot(221)
# plt.imshow(im_src,cmap='gray')
# plt.title('Source image')

# plt.subplot(222)
# plt.imshow(im_out,cmap='gray')
# plt.title('Corrected image')

# print (im_out.shape)
# cv2.imshow('img',im_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

RGB_im_out = np.float32(RGB_im_out) / 255.0
gx = cv2.Sobel(RGB_im_out, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(RGB_im_out, cv2.CV_32F, 0, 1, ksize=1)

mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
plt.figure(figsize=(10,8))
plt.subplot(231)
plt.imshow(RGB_im_out)
plt.title('extarted image')
plt.subplot(232)
plt.imshow(gx)
plt.title('sobelx applied')

plt.subplot(233)
plt.imshow(gy)
plt.title('sobely applied')

plt.subplot(234)
plt.imshow(mag)
plt.title('polar magnitude')

plt.subplot(235)
plt.imshow(angle)
plt.title('polar angle')


