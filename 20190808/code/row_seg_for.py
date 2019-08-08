import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

from PIL import Image, ImageDraw

seg_PATH = 'C:\\Users\\ialab\\Desktop\\Hanja_DKU-master\\seg_part1\\'
PATH = 'C:\\Users\\ialab\\Desktop\\Hanja_DKU-master\\sample\\sample01_ex2_10\\'

for j in range(19):
    if j % 2 == 0: continue
    data = pd.read_csv('seg_4_%d.csv' %j, header=None)
    n_row, n_col = data.shape

    #print(n_row, n_col)

    avg_col = list(data.mean(axis=0))
    # 평균값을 int로 변환
    avg_col = list(map(int, avg_col))
    avg_row = data.mean(axis=1)
    avg_row = list(map(int, avg_row))
    #print(avg_col)

    #plt.plot(avg_col)
    #plt.show()
    # threshold 사용 인자 = array, threshmin, threshmax, newval
    # return 값 = array
    threshold = np.clip(avg_row, 253, 254)
    #print(threshold)
    #plt.plot(threshold)
    #plt.show()

    c_list = []

    for i in range(len(threshold) - 1):
        if threshold[i] != threshold[(i+1)]:
            c_list.append(i)

    print(c_list)

    # data.iloc[:, 0:288]
    for i in range(len(c_list)-1):
        seg = data[c_list[(i)]: c_list[(i+1)]]
        seg_array = seg.values.astype('int')
        image = Image.fromarray(seg_array).convert('RGB')
        #image.show()
        image.save(seg_PATH + 'seg_4_%d_%d.jpg' %(j,i))

    '''
    test = data.iloc[:, c_list[1]:c_list[2]]
    test2 = data.iloc[:, c_list[3]:c_list[4]]
    test3 = data.iloc[:, c_list[5]:c_list[6]]
    test4 = data.iloc[:, c_list[7]:c_list[8]]
    test5 = data.iloc[:, c_list[9]:c_list[10]]
    test6 = data.iloc[:, c_list[11]:c_list[12]]
    test7 = data.iloc[:, c_list[13]:c_list[14]]
    test8 = data.iloc[:, c_list[15]:c_list[16]]
    test9 = data.iloc[:, c_list[17]:c_list[18]]
    test10 = data.iloc[:, c_list[19]:c_list[20]]
    
    data_np_array = test.values.astype('int')
    data_np_array2 = test2.values.astype('int')
    data_np_array3 = test3.values.astype('int')
    data_np_array4 = test4.values.astype('int')
    data_np_array5 = test5.values.astype('int')
    data_np_array6 = test6.values.astype('int')
    data_np_array7 = test7.values.astype('int')
    data_np_array8 = test8.values.astype('int')
    data_np_array9 = test9.values.astype('int')
    data_np_array10 = test10.values.astype('int')
    #print(data_np_array)
    #print(A)
    #to_img = genfromtxt(A, delimiter=',').astype('int')
    #print(to_img)
    img = Image.fromarray(data_np_array).convert('RGB')
    img2 = Image.fromarray(data_np_array2).convert('RGB')
    img3 = Image.fromarray(data_np_array3).convert('RGB')
    img4 = Image.fromarray(data_np_array4).convert('RGB')
    img5 = Image.fromarray(data_np_array5).convert('RGB')
    img6 = Image.fromarray(data_np_array6).convert('RGB')
    img7 = Image.fromarray(data_np_array7).convert('RGB')
    img8 = Image.fromarray(data_np_array8).convert('RGB')
    img9 = Image.fromarray(data_np_array9).convert('RGB')
    img10 = Image.fromarray(data_np_array10).convert('RGB')
    
    
    img.show()
    img2.show()
    img3.show()
    img4.show()
    img5.show()
    img6.show()
    img7.show()
    img8.show()
    img9.show()
    img10.show()
    
    PATH = 'C:\\Git\\hanja\\segment\\sample01_ex1_1'
    
    img.save(seg_PATH + 'sample01_ex2_01.jpg')
    img2.save(seg_PATH + 'sample01_ex2_02.jpg')
    img3.save(seg_PATH + 'sample01_ex2_03.jpg')
    img4.save(seg_PATH + 'sample01_ex2_04.jpg')
    img5.save(seg_PATH + 'sample01_ex2_05.jpg')
    img6.save(seg_PATH + 'sample01_ex2_06.jpg')
    img7.save(seg_PATH + 'sample01_ex2_07.jpg')
    img8.save(seg_PATH + 'sample01_ex2_08.jpg')
    img9.save(seg_PATH + 'sample01_ex2_09.jpg')
    img10.save(seg_PATH + 'sample01_ex2_10.jpg')
    '''