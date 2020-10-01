# This file is for the purpose of providing a gui for my Face Recognition System.
# Version: 1.0.0
# @Author: Malay Bhavsar

# Importing the module.
import tkinter as tk
from tkinter import ttk
import FRS_v1

pt = FRS_v1.leo_frs()

def start_cap():
    master = tk.Tk()
    master.title("LEO-Capture")
    def mquit():
        pt.capture_start(0,name.get())
        pt.train()
        master.destroy()
    ttk.Label(master, 
            text="Name").grid(row=1, pady=10,padx=10)
    name = tk.Entry(master)
    name.grid(row=1,column=1, pady=10,padx=10)
    ttk.Button(master, 
            text='Ok', 
            command=mquit).grid(row=2, column=1, sticky=tk.W, pady=10,padx=10)
    master.mainloop()
    

def import_img():
    master = tk.Tk()
    master.title("LEO-Import")
    def mquit():
        pt.import_img(path.get(),name.get())
        pt.train()
        master.destroy()
    ttk.Label(master, 
            text="Path").grid(row=1, pady=10,padx=10)
    path = tk.Entry(master)
    path.grid(row=1,column=1, pady=10,padx=10)
    ttk.Label(master, 
            text="Name").grid(row=2, pady=10,padx=10)
    name = tk.Entry(master)
    name.grid(row=2,column=1, pady=10,padx=10)
    ttk.Button(master, 
            text='Ok', 
            command=mquit).grid(row=6, column=1, sticky=tk.W, pady=10,padx=10)
    master.mainloop()

def predict():
    pt.predict(0)

# Creating the class..
win = tk.Tk()
win.title("LEO-Face-Recognition")
win.geometry("250x150")
# Adding the button.
b_cap = ttk.Button(win,text="Start Capture",command=start_cap,width=15).grid(row=1,padx=60,pady=10)
b_import = ttk.Button(win,text="Import Images",command=import_img,width=15).grid(row=3,padx=60,pady=10)
b_predict = ttk.Button(win,text="Predict",command=predict,width=15).grid(row=5,padx=60,pady=10)

win.mainloop()