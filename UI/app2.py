import tkinter as tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import joblib
import matplotlib.pyplot as plt
import pickle
import numpy as np 
import os
import tensorflow as tf
import nibabel as nib
from scipy import ndimage
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image


def input_img():
    
    filename = askopenfilename() 
    return filename



def read_nifti_file(filepath):
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
   
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
   
    img = ndimage.rotate(img, 90, reshape=False)
   
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_scan(path):
    volume = read_nifti_file(path)
    volume = normalize(volume)
   
    volume = resize_volume(volume)
    return volume
def model(image_path):
    
    loaded_model = joblib.load("3d_model.joblib")
    loaded_model.load_weights("3d_image_classification.h5")
    processed_image=process_scan(image_path) 
    prediction = loaded_model.predict(np.expand_dims(processed_image, axis=0))[0]

    #print(prediction[0])
    scores = [1 - prediction[0], prediction[0]]

    out=max(scores)

    class_names = ["Normal","Cancer"]
    for score, name in zip(scores, class_names):
        if(score == out):
            res=""
            return(
                "the model is %.2f percent confident that CT scan is %s"
                % ((100 * score), name)
            )
            #print(res)


def acc():
    image_path =r"D:\dataset\LungCancerDetection\model_acc_3d.png"
    image = tf.keras.preprocessing.image.load_img(image_path)
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.axis("off")
    plt.show()



class App:
    def __init__(self, master):
        self.master = master
        master.title("LUNG CANCER DETECTION USING 3D CNN")
        master.geometry("800x600")
        
        # Create a frame for the title
        title_frame = ttk.Frame(master)
        title_frame.pack(pady=0)
        
        # Create a label for the title
        title_label = ttk.Label(title_frame, text="LUNG CANCER DETECTION USING 3D CNN", font=("Times New Roman", 20, ), anchor="center")
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
        output=model(path_img)
        self.display_text.insert(tk.END,"Predicted output: "+output)


        
    def show_accuracy(self):
        acc()

    def exit(self):
        self.master.destroy()


root = tk.Tk()
app = App(root)
root.mainloop()



# print("Lung cancer detection 2D\n")
# print("select image")
# file_name=input_img()

# print("Output:\n")
# model(file_name)
# #print(result)

