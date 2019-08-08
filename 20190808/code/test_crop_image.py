from tkinter import *
from PIL import ImageTk,Image
import cv2
import cv2 as cv
import numpy as np
import tkinter.filedialog as filedialog

cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0

def save_seg_image():
    global pil_result
    file = filedialog.asksaveasfile(mode='w', defaultextension=".jpg")
    if file:
        pil_img = Image.fromarray(pil_result)
        pil_img.save(file)  # saves the image to the input file name.

def mouse_crop(event, x, y, flags, param):
    global oriImage
    global pil_result
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping

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

        if len(refPoint) == 2:  # when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv2.imshow("Cropped", roi)
            pil_result = roi

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
    cv2.setMouseCallback("image", mouse_crop)



root = Tk()


btn = Button(root, text="Crop an image", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn = Button(root, text="Save an segmented image", command=save_seg_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

root.mainloop()
cv2.waitKey(1)

# close all open windows
cv2.destroyAllWindows()
