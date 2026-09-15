# Devanagari Character Recognizer

A desktop Devanagari character recognizer written in modern C++ with OpenCV. Draw a character in an interactive canvas and the application runs a TensorFlow model through OpenCV DNN to predict its class and confidence.

## About

This project demonstrates an end-to-end computer vision workflow for
Devanagari handwriting: interactive image capture, region extraction,
preprocessing, TensorFlow model inference, and visual feedback. It is intended
as an educational and experimental desktop application for Indic-script OCR.

## Features

- Interactive OpenCV drawing canvas
- Automatic bounding-box extraction around the stroke
- 32×32 preprocessing and TensorFlow inference through OpenCV DNN
- Prediction class and confidence rendered on the canvas
- Configurable model path (no machine-specific paths)
- CMake-based build with compiler warnings enabled

## Project structure

```text
.
├── src/main.cpp                    # Interactive application and inference flow
├── models/devanagari_detection.pb  # TensorFlow frozen graph
├── docs/                            # Screenshots and project documentation
├── CMakeLists.txt
└── README.md
```

## Requirements

- C++17 compiler
- CMake 3.16 or newer
- OpenCV 4.x with `core`, `highgui`, `imgproc`, and `dnn` modules
- A compatible TensorFlow frozen graph (`.pb`)

## Build

```bash
git clone https://github.com/utsab345/devanagari-character-recognizer-cpp.git
cd devanagari-character-recognizer-cpp
cmake -S . -B build
cmake --build build --config Release
```

If OpenCV is installed in a non-standard location, provide its CMake package path:

```bash
cmake -S . -B build -DOpenCV_DIR=/path/to/opencv4/lib/cmake/opencv4
```

## Run

From the repository root:

```bash
./build/devanagari_recognizer
```

The default model is `models/devanagari_detection.pb`. To use another model:

```bash
./build/devanagari_recognizer /path/to/model.pb
```

### Controls

- Hold the left mouse button and draw a character.
- Release the mouse button to run recognition.
- Press `q` to clear the canvas.
- Press `Esc` to exit.

The latest prediction is also printed to the terminal. A debug snapshot is written to `debug_canvas.png`.

## Screenshots

Examples of the drawing canvas, detected bounding box, and recognition output:

![Recognizer example 1](https://github.com/user-attachments/assets/81faaa77-213a-4b4b-9c35-6528329558ac)

![Recognizer example 2](https://github.com/user-attachments/assets/26e66e94-e651-4492-b362-1b187dc0ab32)

## How inference works

1. The stroke is enclosed in a padded bounding box.
2. The region is converted to grayscale and resized to 32×32 pixels.
3. OpenCV DNN creates a normalized input blob and runs the frozen TensorFlow graph.
4. The highest-scoring output class and confidence are displayed.

## Limitations

- The output class is shown as a numeric label; a class-to-character label map is not included.
- Recognition quality depends on the training data and the supplied TensorFlow graph.
- This is an interactive desktop application and does not provide a web API.

## Contributing

Keep changes focused, build with warnings enabled, and use conventional commits such as `feat:`, `fix:`, `chore:`, and `docs:`.

## Acknowledgements

- OpenCV for image processing, GUI, and DNN inference.
- TensorFlow for the model format and training ecosystem.
