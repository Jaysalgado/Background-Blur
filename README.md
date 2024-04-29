# Background-Blur Application

## Introduction
Background-Blur is an application that uses a convolutional neural network (CNN) based on the Mask R-CNN architecture with a ResNet-50 FPN backbone, implemented using torchvision. This application allows users to interactively select any object in an uploaded image. The selected object remains in focus while the rest of the image is blurred, emphasizing the selected object.

## Features
- Interactive GUI built with Tkinter.
- Utilizes pretrained CNN models from torchvision to detect objects.
- Allows users to adjust the intensity of the blur.
- Option to save the modified image.

## How to run

### Prerequisites

Make sure you have Python installed. If you're on Windows, make sure Python is added to PATH.

### Windows

Make a virtual environment:

`py -m venv env`

Start the virtual environment:

`activate`

Install the dependencies

`pip install -r requirements.txt`

Run the application

`py main.py`

### Mac

Make a virtual environment:

`python3.11 -m venv env`

Start the virtual environment:

`source env/bin/activate`

Install the dependencies

`pip3 install -r requirements.txt`

Run the application

`python3.11 main.py`

## How to use

1. Select an image using the button
2. Use the slider to determine how much the background should be blurred
3. Click on a subject in the image to blur the background around it
4. If you would like to save the image, a new window will open once a subject is clicked, and there is a button to save the blurred image
