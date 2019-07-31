import numpy as np
import cv2
import csv


seg_PATH = 'C:\\Git\\hanja\\segment\\'

# 이미지 로드
img1 = cv2.imread(seg_PATH + 'sample01_ex2_01.jpg', cv2.IMREAD_UNCHANGED)
img2 = cv2.imread(seg_PATH + 'sample01_ex2_02.jpg', cv2.IMREAD_UNCHANGED)
img3 = cv2.imread(seg_PATH + 'sample01_ex2_03.jpg', cv2.IMREAD_UNCHANGED)
img4 = cv2.imread(seg_PATH + 'sample01_ex2_04.jpg', cv2.IMREAD_UNCHANGED)
img5 = cv2.imread(seg_PATH + 'sample01_ex2_05.jpg', cv2.IMREAD_UNCHANGED)
img6 = cv2.imread(seg_PATH + 'sample01_ex2_06.jpg', cv2.IMREAD_UNCHANGED)
img7 = cv2.imread(seg_PATH + 'sample01_ex2_07.jpg', cv2.IMREAD_UNCHANGED)
img8 = cv2.imread(seg_PATH + 'sample01_ex2_08.jpg', cv2.IMREAD_UNCHANGED)
img9 = cv2.imread(seg_PATH + 'sample01_ex2_09.jpg', cv2.IMREAD_UNCHANGED)
img10 = cv2.imread(seg_PATH + 'sample01_ex2_10.jpg', cv2.IMREAD_UNCHANGED)

# 이미지 그레이 스케일 화
dst1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
dst2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
dst3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
dst4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
dst5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
dst6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
dst7 = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY)
dst8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
dst9 = cv2.cvtColor(img9, cv2.COLOR_BGR2GRAY)
dst10 = cv2.cvtColor(img10, cv2.COLOR_BGR2GRAY)

# threshold 적용 adaptive 방식,(이미지, 임계값 이상일 경우 바꿀 최대값(255 = 흰색), 가우시안 / 평균,
# THRESH_BINARY/THRESH_BINARY_INV = 검출 물체 검은색/흰색, 임계값 계산시 사용되는 블럭 크기, 평균/가중평균에서 뺼값)
#img_threshold = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 5)
#img_threshold = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5)

# 가우시안 필터
img_gaussian1 = cv2.GaussianBlur(dst1, (9, 9), 0)
img_gaussian2 = cv2.GaussianBlur(dst2, (9,9), 0)
img_gaussian3 = cv2.GaussianBlur(dst3, (9, 9), 0)
img_gaussian4 = cv2.GaussianBlur(dst4, (9,9), 0)
img_gaussian5 = cv2.GaussianBlur(dst5, (9, 9), 0)
img_gaussian6 = cv2.GaussianBlur(dst6, (9,9), 0)
img_gaussian7 = cv2.GaussianBlur(dst7, (9, 9), 0)
img_gaussian8 = cv2.GaussianBlur(dst8, (9,9), 0)
img_gaussian9 = cv2.GaussianBlur(dst9, (9, 9), 0)
img_gaussian10 = cv2.GaussianBlur(dst10, (9,9), 0)

# otsu's Binarization
ret1, img_otsu1 = cv2.threshold(img_gaussian1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret2, img_otsu2 = cv2.threshold(img_gaussian2, 0 , 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret3, img_otsu3 = cv2.threshold(img_gaussian3, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret4, img_otsu4 = cv2.threshold(img_gaussian4, 0 , 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret5, img_otsu5 = cv2.threshold(img_gaussian5, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret6, img_otsu6 = cv2.threshold(img_gaussian6, 0 , 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret7, img_otsu7 = cv2.threshold(img_gaussian7, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret8, img_otsu8 = cv2.threshold(img_gaussian8, 0 , 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret9, img_otsu9 = cv2.threshold(img_gaussian9, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret10, img_otsu10 = cv2.threshold(img_gaussian10, 0 , 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#Prob_hough(img_otsu, 1500, 1000, 500)
# 이미지 파일 생성
#cv2.imwrite('img_otsu(5,5).jpg', img_otsu)
#cv2.imwrite('img_gaussian_filter.jpg', img_gaussian)

# 이미지 보기
#cv2.imshow('test2', img_otsu2)
#cv2.imshow('test', img_otsu)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


test = np.array(img_otsu1)

cvs_file = open(seg_PATH + 'sample01_ex2_01.csv', 'w', newline='')
cvs_writer = csv.writer(cvs_file)
for row in test:
    cvs_writer.writerow(row)

cvs_file.close()

test = np.array(img_otsu2)

cvs_file = open(seg_PATH + 'sample01_ex2_02.csv', 'w', newline='')
cvs_writer = csv.writer(cvs_file)
for row in test:
    cvs_writer.writerow(row)

cvs_file.close()

test = np.array(img_otsu3)

cvs_file = open(seg_PATH + 'sample01_ex2_03.csv', 'w', newline='')
cvs_writer = csv.writer(cvs_file)
for row in test:
    cvs_writer.writerow(row)

cvs_file.close()

test = np.array(img_otsu4)

cvs_file = open(seg_PATH + 'sample01_ex2_04.csv', 'w', newline='')
cvs_writer = csv.writer(cvs_file)
for row in test:
    cvs_writer.writerow(row)

cvs_file.close()

test = np.array(img_otsu5)

cvs_file = open(seg_PATH + 'sample01_ex2_05.csv', 'w', newline='')
cvs_writer = csv.writer(cvs_file)
for row in test:
    cvs_writer.writerow(row)

cvs_file.close()

test = np.array(img_otsu6)

cvs_file = open(seg_PATH + 'sample01_ex2_06.csv', 'w', newline='')
cvs_writer = csv.writer(cvs_file)
for row in test:
    cvs_writer.writerow(row)

cvs_file.close()


test = np.array(img_otsu7)

cvs_file = open(seg_PATH + 'sample01_ex2_07.csv', 'w', newline='')
cvs_writer = csv.writer(cvs_file)
for row in test:
    cvs_writer.writerow(row)

cvs_file.close()

test = np.array(img_otsu8)

cvs_file = open(seg_PATH + 'sample01_ex2_08.csv', 'w', newline='')
cvs_writer = csv.writer(cvs_file)
for row in test:
    cvs_writer.writerow(row)

cvs_file.close()

test = np.array(img_otsu9)

cvs_file = open(seg_PATH + 'sample01_ex2_09.csv', 'w', newline='')
cvs_writer = csv.writer(cvs_file)
for row in test:
    cvs_writer.writerow(row)

cvs_file.close()

test = np.array(img_otsu10)

cvs_file = open(seg_PATH + 'sample01_ex2_10.csv', 'w', newline='')
cvs_writer = csv.writer(cvs_file)
for row in test:
    cvs_writer.writerow(row)

cvs_file.close()
