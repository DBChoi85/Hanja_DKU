from tkinter import *
from PIL import ImageTk,Image
import cv2
import cv2 as cv
import numpy as np
import tkinter.filedialog as filedialog
import json


cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0


def save_seg_image():
    global pil_result
    file = filedialog.asksaveasfile(mode='w', defaultextension=".jpg")
    if file:
        pil_img = Image.fromarray(pil_result)
        pil_img.save(file)  # saves the image to the input file name.


def crop_area(x1, x2, y1, y2):
    global oriImage
    global pil_result
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping

    refPoint = [(x1, y1), (x2, y2)]
    roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
    cv2.imshow("Cropped", roi)
    pil_result = roi

def bboxSelet(event):
    global area

    widget = event.widget
    selection=widget.curselection()#선택된 항목을 튜플로 변환
    picked = widget.get(selection[0])
    bbox_item = picked.split(" ")
    bbox_idx = int(bbox_item[1])
    ##인덱스로 활용하기위해 맨끝의 번호 받음

    c1 = area[bbox_idx]
    print (c1)
    x1 = int(c1[1])
    y1 = int(c1[0])
    x2 = int(c1[3])
    y2 = int(c1[2])
    crop_area(x1, x2, y1, y2)

def load_bbox_json():
    global oriImage
    global area

    path = filedialog.askopenfilename()
    Lb = Listbox(root)
    if len(path) > 0:
        print(path)
        area = []
        with open(path) as json_file:
            data = json.load(json_file)
            for index, p in enumerate(data['bbox[x1,y1,x2,y2]']):
                Lb.insert(index, 'BoundBox '+str(index))#리스트 항목 추가
                area.append(p)
            # print(area)

    Lb.bind('<<ListboxSelect>>', bboxSelet)
    #bind -> 어떤 이벤트가 작동했을때 함수 적용시킴;; <<ListboxSelect>>은 리스트에 있는 항목 선택시 발생되는 이벤트
    Lb.pack()
    if area != None:
       c1 = area[0]
       x1 = int(c1[1])
       y1 = int(c1[0])
       x2 = int(c1[3])
       y2 = int(c1[2])
       crop_area(x1, x2, y1, y2)


def select_image():
    global oriImage

    path = filedialog.askopenfilename()

    # ensure a file path was selected
    if len(path) > 0:
        print(path)
        oriImage = cv.imread(path)
        oriImage = cv.cvtColor(oriImage, cv.COLOR_BGR2GRAY)

        if not cropping:
            cv2.imshow("image", oriImage)

        elif cropping:
            cv2.rectangle(oriImage, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("image", oriImage)

    cv2.namedWindow("image")

    # read a json file
    # extract a set of areas
    # select the first area from the set
    # crop the area selected  by a function crop_area()

    area = []
    with open('bboxtojson.json') as json_file:
        data = json.load(json_file)
        for p in data['bbox[x1,y1,x2,y2]']:
            area.append(p)
            #print(area)

    c1 = area[0]
    x1 = int(c1[1])
    y1 = int(c1[0])
    x2 = int(c1[3])
    y2 = int(c1[2])
    crop_area(x1, x2, y1, y2)

"""
    x1 = 604
    y1 = 7
    x2 = 731
    y2 = 289
"""


root = Tk()

btn = Button(root, text="1. Crop a chinese text image", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn = Button(root, text="2. Save a segmented image", command=save_seg_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn = Button(root, text="3. Load bbox JSON", command=load_bbox_json)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")


root.mainloop()
cv2.waitKey(1)

# close all open windows
cv2.destroyAllWindows()