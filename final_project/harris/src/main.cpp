#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include "HarrisCorner.h"

using namespace cv;
using namespace std;

typedef std::chrono::high_resolution_clock Clock;

/**
 * Run a serial implementation of the detector
 */
void run_cpu(Mat input_image, const char* output_file) {
    // Get the size
    unsigned int height = input_image.rows;
    unsigned int  width = input_image.cols;
    
    // New mat has height/width the same as the old
    Mat image_corners = Mat::zeros(height, width, CV_32FC1);
    
    // Create a place to dump the detected keypoints
    std::vector<cv::KeyPoint> kps;
    
    // This is the actual detection
    HarrisCorner* cpu = HarrisCorner::create(0.04f, 1e5f);
    auto t1 = Clock::now();                 // Make sure to start timing
    cpu->detect(input_image, kps);
    auto t2 = Clock::now();                 // End timing
    cout << "Time to find CPU: "            // Print results
         << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
         << " ms" << std::endl;

    // Draw results
    drawKeypoints(input_image, kps, image_corners, Scalar( 0, 0, 255));
    cout << "writing output image " << output_file << endl;
    imwrite (output_file, image_corners);
}

int main( int argc, const char** argv ) {
    // arg 1: Input image 
    // arg 2: output file name
        
    // Check inputs
    if (argc != 3){
        cout << "Incorrect number of inputs" << endl;
        cout << argv[0] << " <input file> <output file name>" << endl;
        return -1;
    }        
        
    // Read input image from argument in black and white
    Mat input_image = imread(argv[1], IMREAD_GRAYSCALE);

    if (input_image.empty()){
        cout << "Image cannot be loaded..!!" << endl;
        return -1;
    }
    
    run_cpu(input_image, argv[2]);

    return 0;
    
}
