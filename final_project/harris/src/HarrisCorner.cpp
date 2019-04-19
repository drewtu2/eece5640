#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <omp.h>

#include "HarrisCorner.h"
#include "utils.h"

using std::string;
using cv::Mat;
using std::endl;
using std::cout;

HarrisCorner* HarrisCorner::create(float k, float threshold) {
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
    Mat image_mat = image.getMat();
    image_mat.convertTo(image_mat, CV_32FC1);       // Convert to 32_FC1 to avoid overflow
    
    Mat ix = Mat::zeros(image_mat.size(), CV_32FC1);
    Mat iy = Mat::zeros(image_mat.size(), CV_32FC1);
    
    int width = image_mat.cols;
    int height = image_mat.rows;

    this->calculate_gradients(
            (float*)ix.data, (float*)iy.data, 
            (float*)image_mat.data, 
            width, height);
    
    // Multiply matrices together
    Mat ix2 = element_mul(ix, ix);
    Mat iy2 = element_mul(iy, iy);
    Mat ixy = element_mul(ix, iy);

    // Harris Corner Response Matrix: 
    // [ix2, ixy;
    // ixy, iy2]

    // Trace is added together the main diagonal
    Mat itrace = element_add(ix2, iy2);

    // Determinant at each pixel is the difference of the two diagonals
    //Mat idet = ix2.mul(iy2) - ixy.mul(ixy);
    Mat ix2y2   = element_mul(ix2, iy2);
    Mat ixy2    = element_mul(ixy, ixy);
    Mat idet    = element_subtract(ix2y2, ixy2);

    // Response
    Mat itrace2 = element_mul(itrace, itrace);
    itrace2 = scalar_mul(itrace2, this->k);
    Mat response = abs(element_subtract(idet, itrace2));

    //Mat writeableResponse;
    //response.convertTo(writeableResponse, CV_8UC1, 255.0);
    //print_bounds(response, "Response ");
    //print_bounds(writeableResponse, "Response ");
    //cv::imwrite ("response.jpg", writeableResponse);

    HarrisCorner::thresholding(keypoints, response);
}

void HarrisCorner::thresholding(std::vector<KeyPoint>& keypoints, InputArray input) {
    Mat response = input.getMat();
    int width = response.cols;
    int height = response.rows;
    int x, y;
    float response_value; 
    // Iterate over the entire image
    //#pragma omp parallel for
    for(int ii = 0; ii < width*height; ++ii) {
        x = ii % width;
        y = ii / width;
        response_value = response.at<float>(y, x);
        if (response_value > this->corner_response_threshold) {
            //#pragma omp critical
            {
                keypoints.push_back(KeyPoint((float)x, (float)y, 3));
            }
        }
    }

}
void HarrisCorner::calculate_gradients(float* ix, float* iy, float* input, int width, int height) {

    int x, y;
    int x_minus_one, x_plus_one;
    int y_minus_one, y_plus_one;
    float dx, dy;

    // Iterate over the entire image
    #pragma omp parallel for
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
