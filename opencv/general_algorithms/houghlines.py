#%%
import cv2
import numpy as np
from matplotlib import pyplot as plt

#######################################
#%%
image = cv2.imread('/home/saivinay/Documents/imgpro/img8.jpg')
img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# plt.imshow())
plt.imshow(image,cmap='gray')
plt.show()

#########################################
#%%
#### using HoughLines
# blur = cv2.bilateralFilter(img,9,75,75)
blur = cv2.GaussianBlur(img,(5,5),0)


# img = cv2.GaussianBlur(img,3)
edges = cv2.Canny(blur,50,200, 3)
# ret, edges = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY)
# cv2.
plt.imshow(edges,cmap='gray')
plt.show()

#if the last argument increases then it finds more larger lines 
lines = cv2.HoughLines(edges,1,np.pi/180,230)
for line in lines:    
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(image,(x1,y1),(x2,y2),(255,0,0),2)

plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.show()

#%%
print(line.shape)

#############################################
#%%

#### using HoughLinesP

blur = cv2.GaussianBlur(img,(5,5),0)


# img = cv2.GaussianBlur(img,3)
edges = cv2.Canny(blur,15,255)
plt.imshow(edges,cmap='gray')
plt.show()

minLineLength = 1
maxLineGap = 10

lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)

for line in lines:    
    for x1,y1,x2,y2 in line:
        # a = np.cos(theta)
        # b = np.sin(theta)
        # x0 = a*rho
        # y0 = b*rho
        # x1 = int(x0 + 1000*(-b))
        # y1 = int(y0 + 1000*(a))
        # x2 = int(x0 - 1000*(-b))
        # y2 = int(y0 - 1000*(a))
        cv2.line(image,(x1,y1),(x2,y2),(255,0,0),4)

plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.show()