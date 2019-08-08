import numpy as np
import cv2 as cv
import cv2

# import the necessary packages
from tkinter import *
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog as filedialog

x_start, y_start, x_end, y_end = 0, 0, 0, 0

def crop_sub():
    global cropping
    global panelB
    if panelB != None:
        cropping = True
        select_image()

def select_sub():
    global cropping
    global path_img
    path_img = None
    cropping = False
    select_image()


def save_seg_image():
    global pil_result
    global panelB
    if panelB != None:
        file = filedialog.asksaveasfile(mode='w', defaultextension=".jpg")
        if file:
            pil_img = Image.fromarray(pil_result)
            #cv2.imshow("img",pil_result)
            pil_img.save(file)  # saves the image to the input file name.



def select_image():
    # grab a reference to the image panels
    global panelA, panelB
    global pil_result
    global path_img
    global oriImage


    # open a file chooser dialog and allow the user to select an input
    # image
    if path_img == None :
        path = filedialog.askopenfilename()
        path_img  = path

    # ensure a file path was selected
    if len(path_img) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        oriImage = cv.imread(path_img)

        if cropping:

            #cv2.rectangle(oriImage, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("image", oriImage)
            cv2.namedWindow("image")
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
        cv2.destroyWindow("image")

        if len(refPoint) == 2:  # when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            #cv2.imshow("Cropped", roi)
            pil_result = roi

            pil_img = Image.fromarray(pil_result)

            edged = ImageTk.PhotoImage(pil_img)
            # update the pannels
            panelB.configure(image=edged)
            panelB.image = edged





# initialize the window toolkit along with the two image panels
root = Tk()
panelA = None
panelB = None
path_img = None
pil_result = None

# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn = Button(root, text="Select an image", command=select_sub)
btn.pack(side="top", anchor = "center",expand="yes", padx="10", pady="10")
btn = Button(root, text="Crop an image", command=crop_sub)
btn.pack(side="top", anchor = "center",expand="yes", padx="10", pady="10")
btn = Button(root, text="Save an segmented image", command=save_seg_image)
btn.pack(side="top", anchor = "center",expand="yes", padx="10", pady="10")

# kick off the GUI
root.mainloop()
