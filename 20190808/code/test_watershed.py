import numpy as np
import cv2 as cv
import cv2

# import the necessary packages
from tkinter import *
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog as filedialog



def save_seg_image():
    global pil_result
    file = filedialog.asksaveasfile(mode='w', defaultextension=".jpg")
    if file:
        # pil_img = Image.fromarray(pil_result)
        pil_result.save(file)  # saves the image to the input file name.


def select_image():
    # grab a reference to the image panels
    global panelA, panelB
    global pil_result
    global path_img

    # open a file chooser dialog and allow the user to select an input
    # image
    path = filedialog.askopenfilename()
    path_img  = path

    # ensure a file path was selected
    if len(path) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it

        img = cv.imread(path)


        pil_img = Image.fromarray(img)
        # pil_result is used for saving
        pil_result = pil_img

        oriImage = cv.imread(path_img)
        cv2.rectangle(oriImage, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.imshow("image", oriImage)
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", mouse_crop)

        pil_img = pil_result

        #pil_gray = Image.fromarray(gray)

        # ...and then to ImageTk format
        #image = ImageTk.PhotoImage(pil_gray)
        edged = ImageTk.PhotoImage(pil_img)

        # if the panels are None, initialize them
        '''
        if panelA is None or panelB is None:
           # the first panel will store our original image
           panelA = Label(image=image)
           panelA.image = image
           panelA.pack(side="left", padx=10, pady=10)
           # while the second panel will store the edge map
           panelB = Label(image=edged)
           panelB.image = edged
           panelB.pack(side="right", padx=10, pady=10)
        # otherwise, update the image panels
        else:
        # update the pannels
           panelA.configure(image=image)
           panelB.configure(image=edged)
           panelA.image = image
           panelB.image = edged
        '''

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


cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0


def mouse_crop(event, x, y, flags, param):

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


def crop_sub():
    global oriImage
    global panelC
    print(path_img)

    oriImage = cv.imread(path_img)
    cv2.rectangle(oriImage, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
    cv2.imshow("image", oriImage)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_crop)



    pil_img = pil_result

    # ...and then to ImageTk format
    #image = ImageTk.PhotoImage(pil_gray)
    edged = ImageTk.PhotoImage(pil_img)







# initialize the window toolkit along with the two image panels
root = Tk()
panelA = None
panelB = None
pil_result = None

# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn = Button(root, text="Select an image", command=select_image)
btn.pack(side="top", anchor = "center",expand="yes", padx="10", pady="10")
btn = Button(root, text="Crop an image", command=crop_sub)
btn.pack(side="top", anchor = "center",expand="yes", padx="10", pady="10")
btn = Button(root, text="Save an segmented image", command=save_seg_image)
btn.pack(side="top", anchor = "center",expand="yes", padx="10", pady="10")

# kick off the GUI
root.mainloop()
