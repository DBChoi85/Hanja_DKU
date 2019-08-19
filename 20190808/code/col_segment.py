import numpy as np
import os

from PIL import Image, ImageDraw


def col_ch(array,array_ori,name,seg_PATH,final=False,j=0):


    if not os.path.isdir(seg_PATH) :
        os.makedirs(seg_PATH)

    img_dir = seg_PATH+'/'+name

    if not os.path.isdir(img_dir) :
        os.makedirs(img_dir)

    #seg_PATH = img_dir

    avg_col = list(np.mean(array, axis=0))
    # 평균값을 int로 변환
    avg_col = list(map(int, avg_col))
    avg_row = list(np.mean(array, axis=1))
    avg_row = list(map(int, avg_row))

    min = 253
    max = 254
    threshold = np.clip(avg_col, min, max)

    c_list = []

    state = 0
    for i in range(len(threshold) - 1):
        if state == 1:
            if threshold[i] == min and threshold[(i + 1)] == max:
                c_list.append(i)
                state = 0
        elif state == 0:
            if threshold[i] == max and threshold[(i + 1)] == min:
                c_list.append(i)
                state = 1

    #print(c_list)

    # data.iloc[:, 0:288]
    count = 0
    for i in range(len(c_list) - 1):
        ind = i * 2
        if ind + 1 > len(c_list) - 1: break
        #print(len(c_list))
        #print(ind + 1)
        try:
            seg = array_ori[:,c_list[ind]: c_list[ind + 1]]
        except TypeError : break
        seg_array = seg.astype(int)
        seg_h,seg_w = seg_array.shape
        if seg_w <=40 :  continue
        image = Image.fromarray(seg_array).convert('RGB')
        # image.show()
        if final == True:
            image.save('C:\\Users\\ialab\\Desktop\\Hanja_DKU-master\\final_result\\seg_3_%d_%d.jpg' % (j, i))
        else :
            image.save(img_dir+'\\' + name+'_%d.jpg' %count)
            count +=1

    return 'col_result'
