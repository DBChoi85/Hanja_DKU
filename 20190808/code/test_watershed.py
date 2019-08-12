import numpy as np
import cv2 as cv
import cv2

# import the necessary packages
from tkinter import *
from tkinter import messagebox
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog as filedialog
import numpy.random.common
import numpy.random.bounded_integers
import numpy.random.entropy

from main_pix import line_seg
import os

seg_PATH = None
seg_PATH_1 = 'C:\\line_seg(block)\\'
seg_PATH_2 = 'C:\\line_seg(hand)\\'

x_start, y_start, x_end, y_end = 0, 0, 0, 0

def crop_sub():
    global cropping
    global panelB
    if panelB != None:
        cropping = True
        select_image()

def select_sub():#목판
    global cropping
    global path_img
    global seg_PATH
    seg_PATH = seg_PATH_1
    path_img = None
    cropping = False
    select_image()

def select_sub_2():
    global cropping
    global path_img
    global seg_PATH
    seg_PATH = seg_PATH_2
    path_img = None
    cropping = False
    select_image()


def save_seg_image():
    global pil_result
    global ori_pil_result
    global panelB
    if panelB != None:
        file = filedialog.asksaveasfile(mode='w', defaultextension=".jpg")
        if file:
            pil_img = Image.fromarray(ori_pil_result)
            #cv2.imshow("img",pil_result)
            pil_img.save(file)  # saves the image to the input file name.



def select_image():
    # grab a reference to the image panels
    global panelA, panelB
    global pil_result
    global path_img
    global oriImage
    global seg_PATH
    global ori_Image
    global categori



    # open a file chooser dialog and allow the user to select an input
    # image
    if path_img == None :
        path = filedialog.askopenfilename()
        path_img  = path


    # ensure a file path was selected
    if len(path_img) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        ori_Image = cv.imread(path_img)
        print(path_img)
        #ori_h = ori_Image.shape[0]
        #ori_w = ori_Image.shape[1]

        if seg_PATH == seg_PATH_1:  # 목판
            oriImage = cv2.resize(ori_Image, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            categori = 1
        elif seg_PATH == seg_PATH_2 :#필사
            oriImage = cv2.resize(ori_Image, dsize=(0, 0), fx=0.25,fy=0.25, interpolation=cv2.INTER_AREA)
            categori = 2


        if cropping:

            #cv2.rectangle(oriImage, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)

            cv2.imshow("image", oriImage)
            cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback("image", mouse_crop)

        else:
            print(cropping)
            #oriImage = cv.imread(path)
            pil_result = oriImage

        pil_img = Image.fromarray(pil_result)

        #pil_gray = Image.fromarray(gray)

        # ...and then to ImageTk format
        #image = ImageTk.PhotoImage(pil_gray)
        edged = ImageTk.PhotoImage(pil_img)

        # if the panels are None, initialize them

        if panelB is None:
            # while the second panel will store the edge map
            panelB = Label(image=edged)
            panelB.image = edged
            panelB.pack(side="right", padx=10, pady=10)
            # otherwise, update the image panels
        else:
            # update the pannels
            panelB.configure(image=edged)
            panelB.image = edged





def mouse_crop(event, x, y, flags, param):
    global panelB
    global pil_result
    global ori_Image
    global categori
    global ori_pil_result
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping
    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    r = cv2.selectROI("image", oriImage, False, False)
    cv2.destroyWindow("image")
    roi = oriImage[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    if categori == 1:  # 목판
        ori_roi = ori_Image[int(r[1])*2:int(r[1] + r[3])*2, int(r[0])*2:int(r[0] + r[2])*2]
    elif categori == 2 :# 목판
        ori_roi = ori_Image[int(r[1])*4:int(r[1] + r[3])*4, int(r[0])*4:int(r[0] + r[2])*4]

    # roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
    # cv2.imshow("Cropped", roi)
    pil_result = roi
    ori_pil_result = ori_roi
    # cv2.imshow("Image2", pil_result)
    # cv2.waitKey(0)

    pil_img = Image.fromarray(pil_result)

    edged = ImageTk.PhotoImage(pil_img)
    # update the pannels
    panelB.configure(image=edged)
    panelB.image = edged

    '''
    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True

    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y

    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False  # cropping is finished

        refPoint = [(x_start, y_start), (x_end, y_end)]
        cv2.destroyWindow("image")
        

        if len(refPoint) == 2:  # when two points were found
            r = cv2.selectROI("image",oriImage,False,False)
            cv2.waitKey(0)
            cv2.destroyWindow("image")
            roi = oriImage[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
            #roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            #cv2.imshow("Cropped", roi)
            pil_result = roi
            #cv2.imshow("Image2", pil_result)
            #cv2.waitKey(0)

            pil_img = Image.fromarray(pil_result)

            edged = ImageTk.PhotoImage(pil_img)
            # update the pannels
            panelB.configure(image=edged)
            panelB.image = edged
            '''



def line_seg_sub():
    global path_img
    global panelB
    global seg_PATH
    if panelB != None:
        line_seg(path_img,seg_PATH)

        messagebox.showinfo("완료", 'Segment Complete')



def open_dir():
    if not os.path.isdir(seg_PATH) :
        os.makedirs(seg_PATH)
    os.startfile(seg_PATH)


# initialize the window toolkit along with the two image panels
root = Tk()
button_frame = Frame(root)
button_frame.pack(side = 'right')

panelA = None
panelB = None
path_img = None
pil_result = None

# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn = Button(button_frame, text="Select an image(목판)", command=select_sub)
btn.pack(side="top", anchor = "center",expand="yes", padx="10", pady="10")
btn = Button(button_frame, text="Select an image(필사)", command=select_sub_2)
btn.pack(side="top", anchor = "center",expand="yes", padx="10", pady="10")
btn = Button(button_frame, text="Line segmented", command=line_seg_sub)
btn.pack(side="top", anchor = "center",expand="yes", padx="10", pady="10")
btn = Button(button_frame, text="Open foder", command=open_dir)
btn.pack(side="top", anchor = "center",expand="yes", padx="10", pady="10")
btn = Button(button_frame, text="Crop an image", command=crop_sub)
btn.pack(side="top", anchor = "center",expand="yes", padx="10", pady="10")
btn = Button(button_frame, text="Save an segmented image", command=save_seg_image)
btn.pack(side="top", anchor = "center",expand="yes", padx="10", pady="10")




# kick off the GUI
root.mainloop()
