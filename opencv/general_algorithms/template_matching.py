#%%
import cv2
import numpy as np
from matplotlib import pyplot as plt

#%%
mainImage = cv2.imread('/home/saivinay/Documents/imgpro/img2.jpg')
mainImage = cv2.cvtColor(mainImage,cv2.COLOR_BGR2RGB)
plt.imshow(mainImage)
plt.show()

#%%
# making template
src_pts = np.array([[335,285],[390,285],[335,340],[390,340]])
dst_pts = np.array([[0,0],[55,0],[0,55],[55,55]])

h, status = cv2.findHomography(src_pts,dst_pts)
template = cv2.warpPerspective(mainImage,h,(55,55))
plt.imshow(template)
plt.show()

#%%
# matching image with template and drawing a rectangle 
# template = cv2.cvtColor(template,cv2.COLOR_RGB2GRAY)
# plt.imshow(template,cmap='gray')
# plt.show()

w,h = template.shape[:-1:]

match = cv2.matchTemplate(mainImage,template,cv2.TM_CCOEFF_NORMED)

threshold = 0.5
location = np.where(match>=threshold)

for pt in zip(*location[::-1]):
    cv2.rectangle(mainImage,pt,(pt[0]+w,pt[1]+h),(0,255,255),1)

# cv2.show('detected',mainImage)
plt.imshow(mainImage)
plt.show()

#%%
tempcod = cv2.imread('img4.jpg')
plt.imshow(tempcod)
plt.show()

orgimage = cv2.imread('img5.jpg')
plt.imshow(orgimage)
plt.show()


for i in np.linspace(0.2,2,10):
    image = cv2.resize(orgimage,(0,0),fx = i,fy = i)
    match = cv2.matchTemplate(orgimage,tempcod,cv2.TM_CCOEFF_NORMED)
    threshold = 1
    (minVal, _, minLoc, _) = cv2.minMaxLoc(match)
    if minVal >= threshold :
        break

location = np.where(match >= threshold)
w,h = (tempcod.shape[:-1:])/i 

for pt in zip(*location[::-1]):
    cv2.rectangle(orgimage,pt,(pt[0]*i + w , pt[1]*i + h),(0,0,255),2)

plt.imshow(orgimage)
plt.show()



#%%
# print (np.linspace(0.2,2,10))
print (match.shape)
# print (template.shape[:-1:])
# print (template.shape[:])
# print (template.shape[:-1])
# print (template.shape[::])
# print (template.shape[::-1])



