# -*- coding: utf-8 -*-

import cv2
import numpy as np

# Load the image
PATH = "C:\\Users\\ialab\\Desktop\\Hanja_DKU-master\\sample\\"
file_name = PATH + 'sample01.jpg'
img = cv2.imread(file_name)

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# smooth the image to avoid noises
gray = cv2.medianBlur(gray, 5)
blur = cv2.GaussianBlur(gray, (7, 7), 0)

ret, mask = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)

image_final = cv2.bitwise_and(gray, gray, mask=mask)

# Apply adaptive threshold
thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

# apply some dilation and erosion to join the gaps
thresh = cv2.dilate(thresh, None, iterations=5)
thresh = cv2.erode(thresh, None, iterations=4)
# Find the contours
# contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

#  morphology를 이용하여 이미지의 노이즈나 hole을 제거
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
dilated = cv2.dilate(thresh, kernel, iterations=1)
contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

index = 0
max_h = 0
max_w = 0
sum_h = 0
sum_w = 0

# For each contour, find the bounding rectangle and draw it
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cropped = image_final[y:y + h, x: x + w]
    s = 'd5_e4_hcrop_' + str(index) + '.jpg'

    h, w = cropped.shape[:2]
    sum_h += h
    sum_w += w
    if h > max_h:
        max_h = h
    if w > max_w:
        max_w = w

    # apply average images to segmented images
    if h > 40 and w < 3000 and w > 40:
        index = index + 1
        cv2.putText(img, str(index), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imwrite(s, cropped)

print("index = ", index)
print("max height = ", max_h, "max_width =", max_w)
print("average height = ", sum_h / index, "average width =", sum_w / index)

# Finally show the image
s2 = 'output_hist.jpg'
cv2.imwrite(s2, img)

