
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
my_w = tk.Tk()

b1 = tk.Button(my_w, text='Upload File',
               width=20, command=lambda: upload_file())
b1.grid(row=2, column=1)


def upload_file():
    global img
    f_types = [('Png Files', '*.png')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    img = Image.open(filename)
    img_resized = img.resize((400, 200))  # new width & height
    img = ImageTk.PhotoImage(img_resized)
    b2 = tk.Button(my_w, image=img)  # using Button
    b2.grid(row=3, column=1)


my_w.mainloop()  # Keep the window open
