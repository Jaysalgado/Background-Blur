
import tkinter as tk
from tkinter import filedialog
from cnn import Vision
from PIL import Image, ImageTk

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title('Image Subject Masking')
        
        # app widgets
        self.button = tk.Button(self, text="Please select an image", command=self.open_image)
        self.button.pack(anchor='center')
        
        self.blur = tk.Scale(self, from_=0, to=15, orient=tk.HORIZONTAL)
        self.blur.pack(anchor='center')
        
        self.canvas = tk.Canvas(self)
        self.canvas.pack(anchor='center', fill=tk.BOTH, expand=True)
        
        self.label = None
        
        # make window get created in the center of the screen
        self.resize_window(self)

        # max image size to display within the app
        self.max_width = 1080
        self.max_height = 720
        
        # computer vision class
        self.vision = Vision()
        
        # prediction variables
        self.image = None
        self.predicted_image = None
        self.prediction = None
        self.ratio = 1
        self.file_path = None
        
    # opens image and displays it in the app
    def open_image(self, event=None):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = Image.open(file_path)
            self.file_path = file_path

            # resize image if necessary
            width, height = self.image.size
            self.ratio = 1
            if width > self.max_width or height > self.max_height:
                self.ratio = min(self.max_width / width, self.max_height / height)
                width = int(width * self.ratio)
                height = int(height * self.ratio)
                resized_image = self.image.resize((width, height), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(resized_image)
            else:
                photo = ImageTk.PhotoImage(self.image)
            
            # display new image
            self.canvas.config(width=width, height=height)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            
            # instructions label
            if self.label == None:
                self.label = tk.Label(self, text="Please click on a subject to blur the image.", anchor='center', font=("Arial", 12))
                self.label.pack()
            
            # resize window to fit image
            self.resize_window(self)
            
            # prevent image from being garbage collected
            self.canvas.image = photo
            
            self.predicted_image, self.prediction = self.vision.load_image(self.file_path)
            
            self.bind("<Button-1>", self.on_canvas_click)
            
    # detects when the canvas is clicked, and gets the x and y of the canvas that's clicked
    def on_canvas_click(self, event):
        if isinstance(event.widget, tk.Canvas):
            # get x and y of mouse cursor on canvas
            canvas = event.widget
            x = canvas.canvasx(event.x)
            y = canvas.canvasy(event.y)
            
            # if the image is larger than the window, make sure x and y are compensated for the image blurring
            if self.ratio < 1:
                x = x/self.ratio
                y = y/self.ratio
                
            # blur image
            blurred_image = self.vision.blur_image((x, y), self.prediction, self.predicted_image, self.blur.get())
            
            # show blurred image in new window if a subject is detected
            if blurred_image:
                self.new_window(blurred_image)
                self.label.config(text="Please click on a subject to blur the image.")
            else:
                self.label.config(text="No subject detected! Please click on another subject.")
                
    # creates a new window with the ability to save the image
    def new_window(self, image):
        image_window = tk.Toplevel(self)
        image_window.title("Blurred Image")

        # resizes image if it's too big
        width, height = image.size
        self.ratio = 1
        if width > self.max_width or height > self.max_height:
            self.ratio = min(self.max_width / width, self.max_height / height)
            width = int(width * self.ratio)
            height = int(height * self.ratio)
            resized_image = image.resize((width, height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(resized_image)
        else:
            photo = ImageTk.PhotoImage(image)

        # display the image in a new window
        image_canvas = tk.Canvas(image_window, width=width, height=height)
        image_canvas.pack()
        image_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        image_canvas.image = photo

        # save the image
        save_button = tk.Button(image_window, text="Save Image", command=lambda: self.save_image(image))
        save_button.pack()
        
    # save image to filename of user's choice
    def save_image(self, image):
        file_path = filedialog.asksaveasfilename(defaultextension=".png")
        if file_path:
            image.save(file_path)
    
    # resize window to accomodate all widgets     
    def resize_window(self, window):
        window.update_idletasks()
        window_width = window.winfo_reqwidth()
        window_height = window.winfo_reqheight()
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x_cordinate = int((screen_width/2) - (window_width/2))
        y_cordinate = int((screen_height/2) - (window_height/2))
        self.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

if __name__ == '__main__':
    app = App()
    app.mainloop()