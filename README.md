# Devanagari Character Recognizer

This project is a simple Devanagari character recognition system built using C++ and OpenCV's Deep Neural Network (DNN) module. The system allows you to draw characters on a canvas and automatically recognizes them using a pre-trained deep learning model.

## Features

- **Canvas Drawing**: Users can draw Devanagari characters on the canvas, which the system captures and processes for recognition.
- **Bounding Box Detection**: The system creates a bounding box around the drawn character.
- **Character Recognition**: The system recognizes the character using a pre-trained TensorFlow model and displays the prediction.
- **Result Visualization**: After recognizing the character, the system shows the corresponding image of the character from a predefined folder.

## How it Works

1. **Drawing**: You can draw a character using the mouse on the canvas.
2. **Bounding Box**: The system creates a bounding box around the drawn character.
3. **Recognition**: Once the mouse is released, the system extracts the region of interest (ROI) and processes it for recognition.
4. **Prediction**: The recognized character is displayed above the bounding box.

## Screenshots

Here are some examples of drawn characters and their bounding boxes:

### Example 1:

![dev1](https://github.com/user-attachments/assets/81faaa77-213a-4b4b-9c35-6528329558ac)

### Example 2:
![dev2](https://github.com/user-attachments/assets/26e66e94-e651-4492-b362-1b187dc0ab32)


## Requirements

- OpenCV 4.x
- C++17 or higher
- TensorFlow model for Devanagari character recognition (`.pb` file)

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/DevanagariCharacterRecognizer.git

2. Build the project using CMake:
   ```bash
   mkdir build   
   cd build
   cmake ..
   make
## Usage
- **Draw Characters**: Use the left mouse button to draw characters on the canvas.
- **Clear Canvas**: Press the `q` key to clear the canvas.
- **Exit**: Press the `Esc` key to exit the application.

## Model
The TensorFlow model (`Devanagari detection.pb`) should be placed in the appropriate directory before running the application. You can download or train the model as per your needs.

## Acknowledgements
- The character recognition model is based on TensorFlow's deep learning framework.
- OpenCV is used for image processing and GUI handling.


