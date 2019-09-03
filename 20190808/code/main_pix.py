from convert import array_load
from col_segment import col_ch
import os
import cv2
import numpy as np

def switch1(x, array_0, array_1, array_2):
    return {
        0: array_0,
        1: array_1,
        2: array_2,
    }.get(x, -1)

def compare_array(array):

    avg_col = list(np.mean(array, axis=0))
    avg_col = list(map(int, avg_col))

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

    # print(c_list)

    # data.iloc[:, 0:288]
    count = 0
    for i in range(len(c_list) - 1):
        ind = i * 2
        if ind + 1 > len(c_list) - 1: break
        # print(len(c_list))
        # print(ind + 1)
        try:
            seg = array[:, c_list[ind]: c_list[ind + 1]]
        except TypeError:
            break
        seg_array = seg.astype(int)
        seg_h, seg_w = seg_array.shape
        if seg_w <= 40:  continue
        count += 1
    return count

def line_seg(path,seg_PATH):

    array_ori, array_original = array_load(path)

    dir, file = os.path.split(path)
    path_split_d = file.split('.')
    file_name = path_split_d[0]
    #print(file_name)

    sort_count = []

    h,w= array_ori.shape
    print(h)
    print(w)
    h1 = h / 5 + h * 0.1
    h1 = int(h1)
    h2 = h / 5 - h * 0.1
    h2 = int(h2)
    array_0 = array_ori[h2:h1, :]
    count_0=compare_array(array_0)
    sort_count.append(count_0)

    h1 = h / 2 + h * 0.1
    h1 = int(h1)
    h2 = h / 2 - h * 0.1
    h2 = int(h2)
    array_1 = array_ori[h2:h1, :]
    count_1 = compare_array(array_1)
    sort_count.append(count_1)

    h1 = (h / 5)*4 + h * 0.1
    h1 = int(h1)
    h2 = (h / 5)*4 - h * 0.1
    h2 = int(h2)
    array_2 = array_ori[h2:h1, :]
    count_2 = compare_array(array_2)
    sort_count.append(count_2)

    array_num = 0
    count = count_0
    for i in range(3):
        count_compare = sort_count[i]
        if count < count_compare :
            count = sort_count[i]
            array_num = i

    print("array_num", array_num)
    print("sort_count",sort_count)
    array = switch1(array_num, array_0, array_1, array_2)
    print("array", array)

    filename = col_ch(array,array_original,file_name,seg_PATH)

