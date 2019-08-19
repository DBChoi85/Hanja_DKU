import tkinter as tk
from PIL import Image, ImageTk

root = tk.Tk()
im = Image.open('2_01.jpg')

def down(event):
    global rect
    global start_x, start_y
    start_x = event.x
    start_y = event.y
    rect = canvas.create_rectangle(start_x, start_y, start_x + 1, start_y + 1)


def draw(event):
    global rect, start_x, start_y, curX, curY
    curX, curY = event.x, event.y
    canvas.coords(rect, start_x, start_y, curX, curY)


def up(event):
    x, y ,xx, yy = start_x, start_y, event.x, event.y

    if start_x > event.x:
        x = event.x
    else:
        x = start_x

    if start_y > event.y:
        y = event.y
    else:
        y = start_y

    if start_x < event.x:
        xx = event.x
    else:
        xx = start_x

    if start_y < event.y:
        yy = event.y
    else:
        yy = start_y

    im2 = im.crop([x, y, xx, yy])
    im2.show()



width, height = im.size
canvas = tk.Canvas(root, width=width, height=height)
tk_im = ImageTk.PhotoImage(im)
canvas.create_image(0, 0, anchor='nw', image=tk_im)
canvas.bind("<ButtonPress-1>", down)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", up)
canvas.pack()

root.mainloop()
