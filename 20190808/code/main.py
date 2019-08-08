import numpy as np
import cv2
import csv
import matplotlib as plt


def hough(img, thr):
    lines = cv2.HoughLines(img, 100, np.pi/180, thr)

    for line in lines:
        r, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a*r
        y0 = b*r

        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*a)
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*a)

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 1)

    cv2.imshow('res', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Prob_hough(img, thr, minLineLength, maxLineGap):
    lines = cv2.HoughLinesP(img, 1, np.pi/180, thr, minLineLength, maxLineGap)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 5)

    cv2.imshow('res', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


PATH = "C:\\Users\\ialab\\Desktop\\Hanja_DKU-master\\sample\\"
PATH2 = 'C:\\Git\\sample\\'

# 이미지 로드
img = cv2.imread('sample01.jpg', cv2.IMREAD_UNCHANGED)
# 이미지 그레이 스케일 화
dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# threshold 적용 adaptive 방식,(이미지, 임계값 이상일 경우 바꿀 최대값(255 = 흰색), 가우시안 / 평균,
# THRESH_BINARY/THRESH_BINARY_INV = 검출 물체 검은색/흰색, 임계값 계산시 사용되는 블럭 크기, 평균/가중평균에서 뺼값)
#img_threshold = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 5)
#img_threshold = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5)

# 가우시안 필터
img_gaussian = cv2.GaussianBlur(dst, (9, 9), 0)

# otsu's Binarization
ret, img_otsu = cv2.threshold(img_gaussian, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


#Prob_hough(img_otsu, 1500, 1000, 500)
# 이미지 파일 생성
#cv2.imwrite('img_otsu(5,5).jpg', img_otsu)
#cv2.imwrite('img_gaussian_filter.jpg', img_gaussian)

# 이미지 보기
#cv2.imshow('adaptive', img_threshold)
#cv2.imshow('test', img_otsu)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

test = np.array(img_otsu)

cvs_file = open(PATH + 'img_otsu.csv', 'w', newline='')
cvs_writer = csv.writer(cvs_file)
for row in test:
    cvs_writer.writerow(row)

cvs_file.close()
