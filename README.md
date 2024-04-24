## What is this?

A simple application to blur images using Python and TorchVision.

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