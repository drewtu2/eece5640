#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <string>

#include "utils.h"

using std::string;
using cv::Mat;
using cv::InputArray;
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

Mat scalar_mul(Mat input_mat, float scalar) {
    int width = input_mat.cols;
    int height = input_mat.rows;
    Mat output_mat = Mat::zeros(input_mat.size(), CV_32FC1);
    float* output = (float*)output_mat.data;
    float* input = (float*)input_mat.data;

    #pragma omp parallel for
    for(int ii = 0; ii < width * height; ++ii) {
        output[ii] = input[ii] * scalar;
    }
    
    return output_mat;
}

Mat scalar_add(Mat input_mat, float scalar) {
    int width = input_mat.cols;
    int height = input_mat.rows;
    Mat output_mat = Mat::zeros(input_mat.size(), CV_32FC1);
    float* output = (float*)output_mat.data;
    float* input = (float*)input_mat.data;

    #pragma omp parallel for
    for(int ii = 0; ii < width * height; ++ii) {
        output[ii] = input[ii] + scalar;
    }

    return output_mat;
}

Mat element_mul(Mat input_mat_1, Mat input_mat_2) {
    int width = input_mat_1.cols;
    int height = input_mat_1.rows;
    Mat output_mat = Mat::zeros(input_mat_1.size(), CV_32FC1);
    float* output = (float*)output_mat.data;
    float* input_1 = (float*)input_mat_1.data;
    float* input_2 = (float*)input_mat_2.data;

    #pragma omp parallel for
    for(int ii = 0; ii < width * height; ++ii) {
        output[ii] = input_1[ii] * input_2[ii];
    }

    return output_mat;
}

Mat element_subtract(Mat input_mat_1, Mat input_mat_2) {
    int width   = input_mat_1.cols;
    int height  = input_mat_1.rows;
    Mat output_mat = Mat::zeros(input_mat_1.size(), CV_32FC1);
    float* output = (float*)output_mat.data;
    float* input_1 = (float*)input_mat_1.data;
    float* input_2 = (float*)input_mat_2.data;

    #pragma omp parallel for
    for(int ii = 0; ii < width * height; ++ii) {
        output[ii] = input_1[ii] - input_2[ii];
    }

    return output_mat;
}

Mat element_add(Mat input_mat_1, Mat input_mat_2) {
    int width   = input_mat_1.cols;
    int height  = input_mat_1.rows;
    Mat output_mat = Mat::zeros(input_mat_1.size(), CV_32FC1);
    float* output = (float*)output_mat.data;
    float* input_1 = (float*)input_mat_1.data;
    float* input_2 = (float*)input_mat_2.data;

    #pragma omp parallel for
    for(int ii = 0; ii < width * height; ++ii) {
        output[ii] = input_1[ii] + input_2[ii];
    }
    
    return output_mat;
}
