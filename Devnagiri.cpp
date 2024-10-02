#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

class DevanagariCharacterRecognizer {
public:
    DevanagariCharacterRecognizer() : drawing(false), lastPoint(-1, -1) {
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

            cvtColor(roi, roi, COLOR_BGR2GRAY);

            resize(roi, roi, Size(32, 32));

            Mat blob = dnn::blobFromImage(roi, 1.0 / 255.0, Size(32, 32), Scalar(0), true, false);

            static dnn::Net net = dnn::readNetFromTensorflow("D:/ML in C++/Devanagari Character Recognition/Devanagari detection.pb");
            net.setInput(blob);

            Mat output = net.forward();

            Point classIdPoint;
            double confidence;
            minMaxLoc(output.reshape(1, 1), nullptr, &confidence, nullptr, &classIdPoint);

            int classId = classIdPoint.x;
            cout << "Predicted class: " << classId << " with confidence: " << confidence << endl;

            Mat resultImage = imread("D:/ML in C++/Devanagari Character Recognition/image/" + to_string(classId) + ".png");
            if (resultImage.empty()) {
                cout << "Result image is empty!" << endl;
                return;
            }

            if (extendedBoundingBox.y - resultImage.rows < 0) {
                cout << "Result image does not fit above the bounding box!" << endl;
                return;
            }

            Mat roiAboveCanvas = canvas(Rect(extendedBoundingBox.x, extendedBoundingBox.y - resultImage.rows, resultImage.cols, resultImage.rows));
            resultImage.copyTo(roiAboveCanvas);

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

int main() {
    DevanagariCharacterRecognizer recognizer;
    recognizer.run();
    return 0;
}
