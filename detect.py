# "import cv2" is used to import the OpenCV library that has various functions and classes for computer vision and image processing tasks.
import cv2

#  "import numpy as np" is used to import the NumPy library that provides support for multi-dimensional arrays, matrices and mathematical functions.
import numpy as np

# "import matplotlib.pyplot as plt" is used to import the pyplot module to create various types of visualizations, such as line plots,scatter plots and more.
import matplotlib.pyplot as plt

# "from deepface import DeepFace" is used to import the DeepFace class from the deepface module of the DeepFace library.
#  DeepFace class is used to perform various facial analysis tasks, such as face recognition, facial attribute analysis, emotion recognition, and face verification.
from deepface import DeepFace

# "import requests" is used to import the requests library in a Python program for making HTTP requests handle responses easily.
import requests

# "from io import BytesIO" is used to import the BytesIO class from the io module in Python. BytesIO class is used to read from or write to binary data in memory.
from io import BytesIO

# "from PIL import Image" is used to import the Image module from the PIL (Python Imaging Library) package in Python.
# PIL is a powerful library for opening, manipulating, and saving many different image file formats.
from PIL import Image

# "import ipywidgets as widgets" is used to import the widgets module from the ipywidgets package in Python.
#  widgets module is used to create and interact with various UI controls and widgets.
import ipywidgets as widgets

# "from IPython.display import display, HTML" is used to import the display and HTML functions from the IPython.display module in Python.
# "display" function is used to show or render objects in the output area of a Jupyter notebook cell.
# The HTML class allows you to display HTML content within the notebook.
from IPython.display import display, HTML

# Function to analyze image and display result

def analyze_image(url):
    response = requests.get(url)                            # Send an HTTP GET request to the specified URL
    img = Image.open(BytesIO(response.content))             # Create an Image object from the response content
    img = np.array(img)                                     # Convert the Image object to a NumPy array

    # Perform emotion analysis

    predictions = DeepFace.analyze(img)                     # Analyze the image using DeepFace library
    predictions_dict = predictions[0]                       # Extract the predictions dictionary from the results
    dominant_emotion = predictions_dict['dominant_emotion'] # Get the dominant emotion from the predictions dictionary

    # Perform face detection

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Create a face cascade classifier object
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                                       # Convert the image to grayscale
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)                                                 # Detect faces in the grayscale image

    # Draw rectangles and labels on the image

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        # Draw a rectangle around each face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
        # Display the dominant emotion near each face
        cv2.putText(img, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5, cv2.LINE_4)

    # Resize the image to a standard size

    img = cv2.resize(img, (800,600))# Adjust the width and height as desired

    # Display the analyzed image with original color

    plt.imshow(img)                 # Display the image using Matplotlib
    plt.axis('off')                 # Turn off the axis labels and ticks
    plt.show()                      # Show the plot


# Function to handle button click event

def on_analyze_button_click(b):

    url = text_url.value.strip()    # Get the URL from the text_url widget
    analyze_image(url)              # Call the analyze_image function with the URL

# Create input text and button widgets

text_url = widgets.Text(description='Image URL:', layout=widgets.Layout(width='50%')) # Create a Text widget for entering the image URL
button_analyze = widgets.Button(description='Analyze', layout=widgets.Layout(width='20%', margin='20px 0 0 0')) # Create a Button widget for triggering the analyze function
output_image = widgets.Output(layout=widgets.Layout(justify_content='center'))  # Create an Output widget for displaying the analyzed image
center_alignment = HTML("<style>.widget-html { text-align: center !important; }</style>") # Create an HTML widget to center-align the widgets


# Assign event handler to the button

button_analyze.on_click(on_analyze_button_click)

# Customize the button style

button_analyze.style.button_color = 'pink'

# Display the widgets

display(center_alignment, text_url, button_analyze, output_image)
