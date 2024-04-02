
import tkinter as tk
from tkinter import filedialog
import os

def BlurImage(event=None):
    filename = filedialog.askopenfilename()
    print('Selected:', filename)
    
    blur_radius = w.get()  # Get the blur radius from the scale widget
    print("blurradius:", blur_radius)
    
    # Construct the command to call the cnn.py script with the selected image filename and blur radius
    cmd = f'python "cnn.py" "{filename}" {blur_radius}'
    
    # Execute the command
    os.system(cmd)

root = tk.Tk()
root.geometry("400x400")

button = tk.Button(root, text="Let's Go", command=BlurImage)
button.pack()

w = tk.Scale(root, from_=0, to=200, orient=tk.HORIZONTAL)
w.pack()

root.mainloop()
