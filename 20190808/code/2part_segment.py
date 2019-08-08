import numpy as np
import pandas as pd
import cv2
import csv
import os

PATH = "C:\\Users\\ialab\\Desktop\\Hanja_DKU-master\\seg_part1\\"
seg_PATH = "C:\\Users\\ialab\\Desktop\\Hanja_DKU-master\\seg_part2\\"

file_name = os.listdir(PATH)

print(file_name)

for i in file_name:
    img = cv2.imread(PATH + i)
    h,w = img.shape[:2]
    if h  >153 :
        cv2.imwrite(seg_PATH + '%s.jpg' % (i), img)