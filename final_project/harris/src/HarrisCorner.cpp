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
    
    double minVal; 
    double maxVal; 

    Mat ix = Mat::zeros(image.size(), CV_8UC1);
    Mat iy = Mat::zeros(image.size(), CV_8UC1);
    Mat image_mat = image.getMat();

    minMaxLoc(ix, &minVal, &maxVal);
    cout << "ix min val : " << minVal << endl;
    cout << "ix max val: " << maxVal << endl;

    this->calculate_gradients(
            (byte*)ix.data, (byte*)iy.data, 
            (byte*)image_mat.data, 
            image_mat.cols, image_mat.rows);
    minMaxLoc(ix, &minVal, &maxVal);
    cout << "ix min val : " << minVal << endl;
    cout << "ix max val: " << maxVal << endl;

    // Multiply matrices together
    cout << "Running .muls" << endl;
    Mat ixx = ix.mul(ix);
    Mat iyy = iy.mul(iy);
    Mat ixy = iy.mul(ix);
    minMaxLoc(ixy, &minVal, &maxVal);
    cout << "ixx min val : " << minVal << endl;
    cout << "ixx max val: " << maxVal << endl;

    // Harris Corner Response Matrix: 
    // [ixx, ixy;
    // ixy, iyy]

    // Trace is added together the main diagonal
    Mat itrace = ixx + iyy;

    // Determinant at each pixel is the difference of the two diagonals
    cout << "Calculating determinatnts" << endl;
    Mat idet = ixx.mul(iyy) - ixy.mul(ixy);
    minMaxLoc(idet, &minVal, &maxVal);
    cout << "det min val : " << minVal << endl;
    cout << "det max val: " << maxVal << endl;

    // Response
    cout << "Calculating response" << endl;
    Mat response = idet - this->k*(itrace.mul(itrace));

    minMaxLoc( response, &minVal, &maxVal);
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

void HarrisCorner::calculate_gradients(byte* ix, byte* iy, byte* input, int width, int height) {

    int x, y;
    int x_minus_one, x_plus_one;
    int y_minus_one, y_plus_one;
    int dx, dy;

    // Iterate over the entire image
    for(int ii = 0; ii < width * height; ++ii) {
        x = ii / width;
        y = ii % width;

        // Break out... 
        if(x <= 1 || y <= 1 || x >= width - 2 || y >= height - 2) {
            continue;
        }

        x_minus_one = y*width + (x - 1);
        x_plus_one = y*width + (x + 1);
        y_minus_one = (y - 1)*width + x;
        y_plus_one = (y + 1)*width + x;

        dx = input[x_minus_one] - input[x_plus_one];
        dy = input[y_minus_one] - input[y_plus_one];
        
        ix[ii] = dx;
        iy[ii] = dy;
    }
}
