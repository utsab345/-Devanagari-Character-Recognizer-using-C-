#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

class DevanagariCharacterRecognizer {
public:
    explicit DevanagariCharacterRecognizer(const string& modelPath)
        : drawing(false), lastPoint(-1, -1), modelPath(modelPath) {
        net = dnn::readNetFromTensorflow(modelPath);
        if (net.empty()) {
            throw runtime_error("Unable to load TensorFlow model: " + modelPath);
        }

        namedWindow("Devanagari Character Recognizer", WINDOW_NORMAL | WINDOW_KEEPRATIO);
        resizeWindow("Devanagari Character Recognizer", canvasSize.width, canvasSize.height);
        setMouseCallback("Devanagari Character Recognizer", &DevanagariCharacterRecognizer::drawCallbackStatic, this);
        canvas = Mat::zeros(canvasSize, CV_8UC3);
    }

    void run() {
        while (true) {
            imshow("Devanagari Character Recognizer", canvas);

            char key = waitKey(1);
            if (key == 'q') {
                clearCanvas();
            } else if (key == 27) { 
                break;
            }

            
            Size currentSize = getWindowSize("Devanagari Character Recognizer");
            if (currentSize != canvasSize) {
                canvasSize = currentSize;
                canvas = Mat::zeros(canvasSize, CV_8UC3);
            }
        }
    }

private:
    Mat canvas;
    Size canvasSize = Size(800, 600); 
    bool drawing;
    Point lastPoint;
    Rect boundingBox;
    string modelPath;
    dnn::Net net;

    vector<Point> points;

    static void drawCallbackStatic(int event, int x, int y, int, void* userdata) {
        static_cast<DevanagariCharacterRecognizer*>(userdata)->drawCallback(event, x, y);
    }

    void drawCallback(int event, int x, int y) {
        if (event == EVENT_LBUTTONDOWN) {
            drawing = true;
            lastPoint = Point(x, y);
            points.clear(); 
            points.push_back(lastPoint);
            boundingBox = Rect(x, y, 1, 1); 
        } else if (event == EVENT_MOUSEMOVE) {
            if (drawing) {
                line(canvas, lastPoint, Point(x, y), Scalar(255, 255, 255), 5, LINE_AA);
                lastPoint = Point(x, y);
                points.push_back(lastPoint);

                boundingBox |= Rect(Point(x, y), Size(1, 1));
            }
        } else if (event == EVENT_LBUTTONUP) {
            drawing = false;
            processROI();
        }
    }

    void processROI() {
        
        Rect extendedBoundingBox = boundingBox;
        extendedBoundingBox.x -= 20; 
        extendedBoundingBox.y -= 20; 
        extendedBoundingBox.width += 40; 
        extendedBoundingBox.height += 40; 

        extendedBoundingBox &= Rect(0, 0, canvas.cols, canvas.rows);

        rectangle(canvas, extendedBoundingBox, Scalar(0, 255, 0), 2);

        if (extendedBoundingBox.area() > 0) {
            Mat roi = canvas(extendedBoundingBox);
            if (roi.empty()) {
                cout << "ROI is empty!" << endl;
                return;
            }

            Mat gray;
            cvtColor(roi, gray, COLOR_BGR2GRAY);

            Mat resized;
            resize(gray, resized, Size(32, 32));

            Mat blob = dnn::blobFromImage(resized, 1.0 / 255.0, Size(32, 32), Scalar(0), true, false);
            net.setInput(blob);

            Mat output = net.forward();

            Point classIdPoint;
            double confidence;
            minMaxLoc(output.reshape(1, 1), nullptr, &confidence, nullptr, &classIdPoint);

            int classId = classIdPoint.x;
            cout << "Predicted class: " << classId << " with confidence: " << confidence << endl;

            const string label = "Class: " + to_string(classId) +
                                 "  Confidence: " + to_string(confidence).substr(0, 6);
            putText(canvas, label, Point(extendedBoundingBox.x, max(24, extendedBoundingBox.y - 8)),
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2, LINE_AA);

            imwrite("debug_canvas.png", canvas);
        }
    }

    void clearCanvas() {
        canvas = Mat::zeros(canvasSize, CV_8UC3);
    }

    Size getWindowSize(const string& windowName) {
        Size size;
        Rect rect = getWindowRect(windowName);
        size.width = rect.width;
        size.height = rect.height;
        return size;
    }

    Rect getWindowRect(const string& windowName) {
        return Rect(0, 0, canvasSize.width, canvasSize.height);
    }
};

int main(int argc, char** argv) {
    const string modelPath = argc > 1 ? argv[1] : "models/devanagari_detection.pb";
    try {
        DevanagariCharacterRecognizer recognizer(modelPath);
        recognizer.run();
    } catch (const exception& error) {
        cerr << "Error: " << error.what() << endl;
        return 1;
    }
    return 0;
}
