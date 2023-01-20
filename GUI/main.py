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

import tkinter as tk

window = Tk()
window.title("MSK Abnormality Detector")


# Model Selection
title = Label(window, text="Welcome to MSK Abnormality Dete",
              font=("Arial Bold", 12))
lbl = Label(window, text="Choose Model", font=("Arial Bold", 12))
lbl.grid(column=0, row=0, sticky='w')

rad1 = Radiobutton(window, text='With CLAHE', value=1)
rad2 = Radiobutton(window, text='Without CLAHE', value=2)

rad1.grid(column=0, row=1, sticky='w')
rad2.grid(column=0, row=2, sticky='w')

# Separator
separator = Label(window, text="--------------------------", padx=5, pady=5)
separator.grid(column=0, row=3, sticky='w')


# File Uploading
lbl = Label(window, text="File Uploading", font=("Arial Bold", 12))
lbl.grid(column=0, row=4, sticky='w')


def upload_file():
    global img
    f_types = [('Png Files', '*.png')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    img = Image.open(filename)
    img_resized = img.resize((100, 100))  # new width & height
    img = ImageTk.PhotoImage(img_resized)
    b2 = tk.Button(window, image=img)  # using Button
    b2.grid(row=1, column=2)


b1 = tk.Button(window, text='Upload File',
               width=20, command=lambda: upload_file())
b1.grid(row=5, column=0, sticky='w')


# Menu Info
def menu_info():
    messagebox.showinfo('What is MSK Abnormailty Detector', 'Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industrys standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.')


menu = Menu(window)
new_item = Menu(menu)
new_item.add_command(label='About', command=menu_info)
menu.add_cascade(label='About', menu=new_item)
window.config(menu=menu)


window.geometry('700x500')
window.mainloop()
