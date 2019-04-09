#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <iostream>
#include "HarrisCorner.h"
#include <string>
#include <cmath>

using std::string;
using cv::Mat;
using std::endl;
using std::cout;

//Usefull debugging function from:
// https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv
string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

void print_bounds(InputArray input, string name) {
    double minVal, maxVal;
    minMaxLoc(input.getMat(), &minVal, &maxVal);
    cout << name << " min val : " << minVal << endl;
    cout << name << " max val: " << maxVal << endl;
}


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
    Mat image_mat = image.getMat();
    image_mat.convertTo(image_mat, CV_32FC1);       // Convert to 32_FC1 to avoid overflow
    
    print_bounds(image_mat, "Image");

    cout << "Image channels: " << image_mat.channels() << endl;
    
    int width = image_mat.cols;
    int height = image_mat.rows;

    this->calculate_gradients(
            (float*)ix.data, (float*)iy.data, 
            (float*)image_mat.data, 
            width, height);
    
    print_bounds(ix, "ix");
    print_bounds(iy, "iy");

    // Multiply matrices together
    cout << "Running .muls" << endl;
    Mat ix2 = ix.mul(ix);
    Mat iy2 = iy.mul(iy);
    Mat ixy = iy.mul(ix);

    print_bounds(ix2, "ix2");
    print_bounds(ixy, "ixy");
    print_bounds(iy2, "iy2");

    cv::imwrite ("ix2.jpg", ix2);
    cv::imwrite ("iy2.jpg", iy2);
    // Harris Corner Response Matrix: 
    // [ix2, ixy;
    // ixy, iy2]

    // Trace is added together the main diagonal
    Mat itrace = ix2 + iy2;

    // Determinant at each pixel is the difference of the two diagonals
    cout << "Calculating determinatnts" << endl;
    Mat idet = ix2.mul(iy2) - ixy.mul(ixy);
    print_bounds(itrace, "itrace");
    print_bounds(ix2.mul(iy2), "ix2y2");
    print_bounds(ixy.mul(ixy), "ixyxy");
    print_bounds(idet, "idet");
    cout << "det type: " << type2str(idet.type()) << endl;

    // Response
    cout << "Calculating response" << endl;
    Mat response = abs(idet - this->k*(itrace.mul(itrace)));

    Mat writeableResponse;
    response.convertTo(writeableResponse, CV_8UC1, 255.0);
    print_bounds(response, "Response ");
    print_bounds(writeableResponse, "Response ");
    cv::imwrite ("response.jpg", writeableResponse);

    cout << "response type: " << type2str(response.type()) << endl;
    cout << "threshold value: " << this->corner_response_threshold << endl;
    int x, y;
    float response_value; 
    // Iterate over the entire image
    for(int ii = 0; ii < width*height; ++ii) {
        x = ii % width;
        y = ii / width;
        response_value = response.at<float>(y, x);
        if (response_value > this->corner_response_threshold) {
            cout << "Point at (" << x << ", " << y << ")" << endl;
            keypoints.push_back(KeyPoint((float)x, (float)y, 3));
        }
    }
    print_bounds(response, "final rsponse");
}

void HarrisCorner::calculate_gradients(float* ix, float* iy, float* input, int width, int height) {

    int x, y;
    int x_minus_one, x_plus_one;
    int y_minus_one, y_plus_one;
    float dx, dy;

    // Iterate over the entire image
    for(int ii = 0; ii < width * height; ++ii) {
        x = ii % width;
        y = ii / width;

        // Break out... 
        if(x <= 1 || y <= 1 || x >= width - 2 || y >= height - 2) {
            continue;
        }

        x_minus_one = y*width + (x - 1);
        x_plus_one = y*width + (x + 1);
        y_minus_one = (y - 1)*width + x;
        y_plus_one = (y + 1)*width + x;

        dx = abs(input[x_minus_one] - input[x_plus_one]);
        dy = abs(input[y_minus_one] - input[y_plus_one]);
        
        ix[ii] = dx;
        iy[ii] = dy;
    }
}

void element_mul(byte* output, byte* input_1, byte* input_2, int width, int height) {
    for(int ii = 0; ii < width * height; ++ii) {
        output[ii] = input_1[ii] * input_2[ii];
    }
}
