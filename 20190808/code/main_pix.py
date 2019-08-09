from convert import array_load
#from convert2 import array_many_load
from row_segment import row_ch
from col_segment import col_ch
import numpy as np
from PIL import ImageTk,Image
import cv2
import cv2 as cv
import os

#path = 'C:\\Users\\ialab\\Desktop\\Hanja_DKU-master\\sample\\sample02.jpg'
#seg_PATH = 'C:\\line_seg\\'

def line_seg(path,seg_PATH):

    array_ori = array_load(path)

    dir, file = os.path.split(path)
    path_split_d = file.split('.')
    file_name = path_split_d[0]
    print(file_name)

    h,w= array_ori.shape
    print(h)
    print(w)
    h1 = h/2+h*0.4
    h1 = int(h1)
    h2 = h/2-h*0.4
    h2 = int(h2)
    #w1 = w/2+w/10
    #w1 = int(w1)
    #w2 = w/2-w/10
    #w2 = int(w2)

    #print("h2 :",h2," h1 :",h1," w2 :",w2," w1 :",w1)

    array = array_ori[h2:h1,:]
    #cv2.imshow("Cropped", array)
    #cv2.waitKey(0)
    #img = Image.fromarray(array)
    #img.save(seg_PATH + 'seg_test_1.jpg')
    #cv2.imwrite()

    '''
    filename = row_ch(array)
    print(filename)
    #filename = col_ch(array)
    filename = array_many_load(filename,'col')
    array_many_load(filename,'row',final = True)
    '''

    filename = col_ch(array,array_ori,file_name,seg_PATH)
    #print(filename)
    ##filename = col_ch(array)
    #filename = array_many_load(filename,'row')
    #array_many_load(filename,'col',final = True)

