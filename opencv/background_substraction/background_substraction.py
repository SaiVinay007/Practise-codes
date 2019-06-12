import cv2
import numpy as np

video_path = "./shutter1.mp4"
image_path = "./background.jpg"

background = cv2.imread(image_path)
background = cv2.cvtColor(background, cv2.COLOR_RGB2YCrCb)
cap = cv2.VideoCapture(video_path)

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)

    diff = background - frame 

    cv2.imshow("diff", diff)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # cv2.imwrite("background.jpg",frame)

cv2.destroyAllWindows()
