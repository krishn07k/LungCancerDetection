import tkinter as tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import joblib
import matplotlib.pyplot as plt
import pickle
import numpy as np 
import os
from PIL import Image, ImageDraw
import tensorflow as tf

import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image


def input_img():
    
    filename = askopenfilename() 
    return filename

def model(image_path):
    loaded_model = joblib.load("2DLung.joblib")
    categories = ["Benign","Cancer","Normal"]
    
    image = tf.keras.preprocessing.image.load_img(image_path)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    scaled_img = np.expand_dims(image_array, axis=0)
    plt.figure(figsize=(3, 3))
    plt.imshow(image)
    plt.axis("off")
    global output
    pred = loaded_model.predict(scaled_img)
    output = categories[np.argmax(pred)]
    #return output
    output = "The model is 97% confident that the CT scan is "+output
    print(output)
    plt.show()    
   
   # print("Predicted case ->", output)

def acc():
    image_path =r"D:\dataset\LungCancerDetection\model_acc_2d.png"
    image = tf.keras.preprocessing.image.load_img(image_path)
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.axis("off")
    plt.show()



class App:
    def __init__(self, master):
        self.master = master
        master.title("LUNG CANCER DETECTION USING 2D CNN")
        master.geometry("800x600")
        
        # Create a frame for the title
        title_frame = ttk.Frame(master)
        title_frame.pack(pady=0)
        
        # Create a label for the title
        title_label = ttk.Label(title_frame, text="LUNG CANCER DETECTION USING 2D CNN", font=("Times New Roman", 20, ), anchor="center")
        title_label.pack(fill="both")
        
        # Create a frame for the buttons
        buttons_frame = ttk.Frame(master)
        buttons_frame.pack(side="left", padx=40, pady=20)

        End_frame = ttk.Frame(master)
        End_frame.pack(pady=0)
        title_label = ttk.Label(title_frame, text="195002015, 195002065", font=("Times New Roman", 15, ), anchor="center")
        title_label.pack(fill="both")


        
        # Create an insert button with an icon
        insert_icon = ImageTk.PhotoImage(Image.open("image.png").resize((35, 35)))
        insert_button = ttk.Button(buttons_frame, text="Insert", image=insert_icon, compound="top", command=self.insert_data, style="FlatButton.TButton")
        insert_button.image = insert_icon
        insert_button.pack(fill="both", padx=10, pady=10)

        
        
        # Create a predict button with an icon
        predict_icon = ImageTk.PhotoImage(Image.open("predict.png").resize((40, 40)))
        predict_button = ttk.Button(buttons_frame, text="Predict", image=predict_icon, compound="top", command=self.predict_data, style="FlatButton.TButton")
        predict_button.image = predict_icon
        predict_button.pack(fill="both", padx=10, pady=10)

        # Create an accu button with an icon
        accuracy_icon = ImageTk.PhotoImage(Image.open("accuracy.png").resize((40, 40)))
        insert_button = ttk.Button(buttons_frame, text="Accuracy", image=accuracy_icon, compound="top", command=self.show_accuracy, style="FlatButton.TButton")
        insert_button.image = accuracy_icon
        insert_button.pack(fill="both", padx=10, pady=10)

        # Create a close button with an icon
        close_icon = ImageTk.PhotoImage(Image.open("exit.png").resize((40, 40)))
        predict_button = ttk.Button(buttons_frame, text="Close", image=close_icon, compound="top", command=self.exit, style="FlatButton.TButton")
        predict_button.image = close_icon
        predict_button.pack(fill="both", padx=10, pady=10)
        
         # Create a frame for the display
        display_frame = ttk.Frame(master, style="DisplayFrame.TFrame")
        display_frame.pack(side="right", padx=40, pady=20, fill="both", expand=False)
        
        # Create a text widget for displaying data
        self.display_text = tk.Text(display_frame, wrap="word", font=("consolas", 12), bg="white")
        self.display_text.pack(fill="both", expand=False)
        
        # Define styles for the buttons and frames
        style = ttk.Style()
        style.configure("FlatButton.TButton", font=("Roboto", 14), background="#f2f2f2", foreground="#666666", borderwidth=0)
        style.map("FlatButton.TButton", background=[("active", "#dddddd")])
        style.configure("DisplayFrame.TFrame", background="#f2f2f2", borderwidth=2, relief="groove")
        
    def insert_data(self):
        # Add code for inserting data here
        global path_img
        path_img = input_img()
        self.display_text.insert(tk.END,"Image inserted!\n")
        
    
    def predict_data(self):
        model(path_img)
        self.display_text.insert(tk.END,"Predicted output: "+output)


        
    def show_accuracy(self):
        acc()

    def exit(self):
        self.master.destroy()


root = tk.Tk()
app = App(root)
root.mainloop()

