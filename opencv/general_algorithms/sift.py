#%%
import cv2
import numpy as np
from matplotlib import pyplot as plt

#%%

image = cv2.imread('/home/saivinay/Documents/imgpro/img2.jpg',0)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(image,None)

out = np.zeros(image.shape)
img = cv2.drawKeypoints(image,kp,out)

plt.imshow(img,cmap='gray')
plt.show()

#%%
print(image)