import tkinter as tk
import cv2
from tkinter.filedialog import askopenfilename, askopenfile
from PIL import Image, ImageTk
from tkinter import (
    filedialog,
    ttk,
)


def upload_file(window):
    global img, display_img
    img_width = img_height = 380
    f_types = [("Png and Jpg Files", "*.png *.jpg")]
    filename = askopenfilename(filetypes=f_types)
    img = Image.open(filename)
    img_resized = img.resize((img_width, img_height))
    display_img = ImageTk.PhotoImage(img_resized)

    img = cv2.imread(filename=filename)
    clear_frame(window)

    label = tk.Label(
        window,
        background="black",
        image=display_img,
        compound="top",
    )
    label.configure(foreground="white")
    label.pack()


def get_img():
    if "img" in globals():
        return img
    return None


def clear_frame(window):
    for widget in window.winfo_children():
        widget.destroy()
