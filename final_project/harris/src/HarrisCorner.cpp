#include <opencv2/core/mat.hpp>
#include <iostream>
#include "HarrisCorner.h"

using cv::Mat;
using std::endl;
using std::cout;


HarrisCorner* HarrisCorner::create(float k, float threshold) {
    return new HarrisCorner(k, threshold);
}

HarrisCorner* HarrisCorner::createOpenMP(float k, float threshold) {
    //TODO: Update this to use an openmp version
    return new HarrisCorner(k, threshold);
}

HarrisCorner::HarrisCorner() {
    this->k = .04f;
    this->corner_response_threshold = 1e5f;
}

HarrisCorner::HarrisCorner(float k, float threshold) {
    this->k = k;
    this->corner_response_threshold = threshold;
}

void HarrisCorner::detect(InputArray image, std::vector<KeyPoint> &keypoints, InputArray mask) {
    // Used the following code as reference
    // http://arrayfire.org/docs/computer_vision_2harris_8cpp-example.htm
    Mat ix = Mat::zeros(image.size(), CV_32FC1);
    Mat iy = Mat::zeros(image.size(), CV_32FC1);

    this->calculate_gradients(ix, iy, image);

    // Multiply matrices together
    cout << "Running .muls" << endl;
    Mat ixx = ix.mul(ix);
    Mat iyy = iy.mul(iy);
    Mat ixy = iy.mul(ix);

    // Harris Corner Response Matrix: 
    // [ixx, ixy;
    // ixy, iyy]

    // Trace is added together the main diagonal
    Mat itrace = ixx + iyy;

    // Determinant at each pixel is the difference of the two diagonals
    cout << "Calculating determinatnts" << endl;
    Mat idet = ixx.mul(iyy) - ixy.mul(ixy);

    // Response
    cout << "Calculating response" << endl;
    Mat response = idet - this->k*(itrace.mul(itrace));

    double minVal; 
    double maxVal; 
    Point minLoc; 
    Point maxLoc;
    minMaxLoc( response, &minVal, &maxVal, &minLoc, &maxLoc );
    cout << "min val : " << minVal << endl;
    cout << "max val: " << maxVal << endl;

    // Extract keyponts from image
    int channels = response.channels();
    int nRows = response.rows;
    int nCols = response.cols * channels;

    if (response.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
    }



    uchar* p;
    for(int i = 0; i < nRows; ++i) {
        // get the row ptr
        p = response.ptr<uchar>(i);
        for(int j = 0; j < nCols; ++j)
        {
            if (p[j] > this->corner_response_threshold) {
                keypoints.push_back(KeyPoint((float)i, (float)j, 3));
            }
        }
    }
}

void HarrisCorner::calculate_gradients(OutputArray ix, OutputArray iy, InputArray input) {
    //TODO - implement this
    ix.create(input.size(), input.type());
    iy.create(input.size(), input.type());

    Mat _ix = ix.getMat();
    Mat _iy = iy.getMat();

    _ix = input.getMat();
    _iy = input.getMat();
}
