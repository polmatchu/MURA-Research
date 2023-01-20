import tkinter as tk
from os import path
from distutils.command.upload import upload
from tkinter import *
from tkinter import messagebox
from tkinter.font import ITALIC
from tkinter.ttk import Progressbar
from tkinter import ttk
from tkinter import filedialog
from tkinter import Menu
from PIL import Image, ImageTk
import os
render = None


def write_slogan():
    # get image and display
    image = tk.PhotoImage(file="./Images/home.png")
    imageLabel.configure(image=image)
    imageLabel.image = image


root = tk.Tk()
frame = tk.Frame(root)
frame.pack()

slogan = tk.Button(frame,
                   text="Hello",
                   command=write_slogan)
slogan.pack(side=tk.LEFT)

imageLabel = tk.Label(frame)
imageLabel.pack(side=tk.LEFT)


def upload_file():
    global img
    f_types = [('Png Files', '*.png')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    img = Image.open(filename)
    img_resized = img.resize((100, 100))  # new width & height
    img = ImageTk.PhotoImage(img_resized)
    b2 = tk.Button(frame, image=img)  # using Button
    b2.pack(side=tk.LEFT)


b1 = tk.Button(frame, text='Upload File',
               width=20, command=lambda: upload_file())
b1.pack(side=tk.LEFT)
root.mainloop()
