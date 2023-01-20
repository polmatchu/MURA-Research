import os
os.system("")
from tkinter import Tk, Text, TOP, BOTH, X, N, LEFT
from tkinter.ttk import Frame, Label, Entry
from os import path
from tkinter import *
from tkinter import messagebox
from tkinter.font import ITALIC
from tkinter.ttk import Progressbar
from tkinter import ttk
from tkinter import filedialog
from tkinter import Menu
from PIL import Image, ImageTk
import textwrap
from image_manipulation import (
    preprocessing_without_clahe,
    preprocessing_with_clahe,
)

from model_builder import (
    load_model,
    classify,
)

from file_picker import *

import os
import sys
import tkinter as tk


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


class Application(Frame):
    def __init__(self):

        self.root = tk.Tk()
        self.root.title("MSK Abnormality Detector")
        super().__init__()

    def run(self):
        self.root.geometry("1280x720")
        self.root.update()
        self.root.minsize(self.root.winfo_width(), self.root.winfo_height())
        self.root.resizable(False, False)
        self.initUI()
        self.root.mainloop()

    def initUI(self):
        self.pack(fill=BOTH, expand=True)

        # Welcome Text
        frame1 = Frame(self)
        frame1.pack(fill=X)
        Welcome = Label(
            frame1,
            text="Welcome to MSK Abnormality Detector",
            width=11,
            font=("Arial Bold", 12),
        )
        Welcome.pack(fill=X, padx=5, pady=10, expand=True)

        # Procedure Text
        procedure_label = Label(
            frame1,
            text="Procedure WITHOUT CLAHE",
            font=("Arial Bold", 11),
            foreground="Red",
        )

        procedure_label.pack(fill=X, padx=5, pady=5, expand=True)

        # Model Selection
        lbl1 = Label(
            frame1,
            text="Choose Model",
            width=11,
            font=("Arial Bold", 12),
            highlightbackground="black",
            highlightthickness=1,
        )
        lbl1.pack(side=LEFT, padx=5, pady=4)

        # Classification Button

        classify_btn = tk.Button(
            frame1,
            text="Classify Image",
            width=20,
            command=lambda: self.classify_input(
                choice, classification_result, probability_result, probability_description, frame_cam
            ),
        )
        classify_btn.pack(side=RIGHT, padx=5, pady=4)

        # Classification Label
        clasify_label = Label(
            frame1, text="Classify", width=11, font=("Arial Bold", 12)
        )
        clasify_label.pack(side=RIGHT, padx=5, pady=4)

        # Radio Buttons
        # Radio Functions

        choice = IntVar()
        choice.set(0)

        # Acutal Buttons
        rad1 = Radiobutton(
            frame1,
            text="Without CLAHE",
            value=0,
            command=lambda: self.update_selected_label(procedure_label, False),
            variable=choice,
        )
        rad1.pack(side=LEFT, padx=15, pady=2)

        rad2 = Radiobutton(
            frame1,
            text="With CLAHE",
            value=1,
            command=lambda: self.update_selected_label(procedure_label, True),
            variable=choice,
        )
        rad2.pack(side=LEFT, padx=15, pady=2)

        # File Uploading
        frame3 = Frame(self)
        frame3.pack(fill=X)

        lbl3 = Label(frame3, text="File Upload",
                     width=11, font=("Arial Bold", 12))
        lbl3.pack(side=LEFT, padx=5, pady=4)

        # File Button

        # Frame Main
        frame_main = Frame(self, pady=30, padx=30)
        frame_main.pack()

        # Frame for Preview Image
        frame_preview = Frame(
            frame_main,
            width=500,
            height=500,
            highlightbackground="black",
            highlightthickness=1,
            pady=50,
            padx=50,
        )
        frame_preview.grid(row=1, column=1)

        # Frame for Cam Image
        frame_cam = Frame(
            frame_main,
            width=500,
            height=500,
            highlightbackground="black",
            highlightthickness=1,
            pady=50,
            padx=50,
        )

        frame_cam.grid(row=1, column=3)

        # Classification Results
        frame_results = Frame(frame_main, pady=10, padx=10)
        frame_results.grid(row=1, column=2)

        classification_result = Label(
            frame_results,
            text="Classification:",
            font=("Arial Bold", 12),
            width=18,
        )
        classification_result.grid(row=1, padx=10, pady=50)

        probability_result = Label(
            frame_results, text="Probability:", font=("Arial Bold", 12), width=18
        )
        probability_result.grid(row=2, padx=10, pady=50)

        note = "IMPORTANT: The software classifies the bone in the radiograph based on probability.\nA probability of greater than 50% will classify it as Abnormal, and anything less than 50% will consider it to be Normal."
        probability_description = Label(
            frame_results,
            text=(textwrap.fill(note, 22)),
            font=("Arial Bold", 8),
            width=22
        )
        probability_description.grid(row=3, padx=10, pady=50)

        upload_btn = tk.Button(
            frame3,
            text="Upload File",
            width=20,
            command=lambda: upload_file(frame_preview),
        )
        upload_btn.pack(side=LEFT, padx=5, pady=4)

    def load_clahe(self):
        global clahe_model
        if not "clahe_model" in globals():
            clahe_model = load_model(clahe=True)

    def load_without_clahe(self):
        global without_clahe_model
        if not "without_clahe_model" in globals():
            without_clahe_model = load_model(clahe=True)

    def update_selected_label(self, label, clahe):
        if clahe:
            label.configure(
                text="Procedure WITH CLAHE", font=("Arial Bold", 11), foreground="Green"
            )
        else:
            label.configure(
                text="Procedure WITHOUT CLAHE",
                font=("Arial Bold", 11),
                foreground="Red",
            )

    def classify_input(self, choice, class_label, probability_label, probability_description, frame_output):
        selection = choice.get()
        selected_img = get_img()

        if not selected_img is None:
            if selection == 1:
                self.load_clahe()
                selected_img = preprocessing_with_clahe(selected_img)
                probability, label, heatmap = classify(
                    clahe_model, selected_img)
            else:
                self.load_without_clahe()
                selected_img = preprocessing_without_clahe(selected_img)
                probability, label, heatmap = classify(
                    without_clahe_model, selected_img
                )
        else:
            return

        if label == 'normal':
            color = 'Green'
        else:
            color = 'Red'

        self.update_frame_img(frame_output, heatmap)
        class_label.configure(
            text=f"Classification:\n{label}",
            font=("Arial Bold", 11),
            foreground=color,
            width=20,
        )
        probability_label.configure(
            text=f'Probability:\n{("{:.6f}".format(probability, 6))}', font=("Arial Bold", 11), width=20
        )

    def update_frame_img(self, frame, img):
        global cam_img
        clear_frame(frame)
        cam_img = ImageTk.PhotoImage(image=img)
        label = tk.Label(
            frame,
            background="black",
            image=cam_img,
            compound="top",
        )
        label.configure(foreground="white")
        label.pack()


def main():
    app = Application()
    app.run()


if __name__ == "__main__":
    main()
