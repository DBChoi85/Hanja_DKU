import numpy as np
import pandas as pd
import cv2
import csv
import os

from col_segment import col_ch
from row_segment import row_ch

seg_PATH = 'C:\\Users\\ialab\\Desktop\\Hanja_DKU-master\\'


def array_many_load(filename, string, final = False):
    # 이미지 로드
    PATH = seg_PATH+filename
    file_list = os.listdir(PATH)
    print(PATH)
    for ind,name in enumerate(file_list):
        img = cv2.imread(PATH +'\\'+name, cv2.IMREAD_UNCHANGED)
        # img2 = cv2.imread('sample01.jpg', cv2.IMREAD_UNCHANGED)
        # 이미지 그레이 스케일 화
        dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # dst2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # threshold 적용 adaptive 방식,(이미지, 임계값 이상일 경우 바꿀 최대값(255 = 흰색), 가우시안 / 평균,
        # THRESH_BINARY/THRESH_BINARY_INV = 검출 물체 검은색/흰색, 임계값 계산시 사용되는 블럭 크기, 평균/가중평균에서 뺼값)
        # img_threshold = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 5)
        # img_threshold = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5)

        # 가우시안 필터
        img_gaussian = cv2.GaussianBlur(dst, (9, 9), 0)
        # img_gaussian2 = cv2.GaussianBlur(dst2, (9,9), 0)

        # otsu's Binarization
        ret, img_otsu = cv2.threshold(img_gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # ret2, img_otsu2 = cv2.threshold(img_gaussian2, 0 , 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Prob_hough(img_otsu, 1500, 1000, 500)
        # 이미지 파일 생성
        # cv2.imwrite('img_otsu(5,5).jpg', img_otsu)
        # cv2.imwrite('img_gaussian_filter.jpg', img_gaussian)

        # 이미지 보기
        # cv2.imshow('test2', img_otsu2)
        # cv2.imshow('test', img_otsu)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        test = np.array(img_otsu)

        if string =='col' : col_ch(test, final, ind)
        elif string =='row' : row_ch(test,final, ind)


        '''
        test2 = np.array(img_otsu2)
    
        cvs_file = open('sample01_ex2.csv', 'w', newline='')
        cvs_writer = csv.writer(cvs_file)
        for row in test2:
            cvs_writer.writerow(row)
    
        cvs_file.close()
        '''
    if string == 'col':
        return 'col_result'
    elif string == 'row':
        return 'row_result'
