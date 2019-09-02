from convert import array_load
from col_segment import col_ch
import os
import cv2
import numpy as np



def line_seg(path,seg_PATH):

    array_ori, array_original = array_load(path)

    dir, file = os.path.split(path)
    path_split_d = file.split('.')
    file_name = path_split_d[0]
    #print(file_name)

    h,w= array_ori.shape
    print(h)
    print(w)
    h1 = h/2+h*0.1
    h1 = int(h1)
    h2 = h/2-h*0.1
    h2 = int(h2)

    array = array_ori[h2:h1,:]


    filename = col_ch(array,array_original,file_name,seg_PATH)

