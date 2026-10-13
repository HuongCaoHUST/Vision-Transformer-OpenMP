#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

int main() {
    cv::Mat img = cv::imread("0010.png");

    if (img.empty()) {
        cerr << "Unable to read the image\n";
        return 1;
    }

    cout << img.rows << " x " << img.cols << " x " << img.channels() << '\n';

    return 0;
}
