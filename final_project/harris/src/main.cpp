#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "HarrisCorner.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>

using namespace cv;
using namespace std;

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
    
    unsigned int height = input_image.rows;
    unsigned int  width = input_image.cols;
    
    HarrisCorner* cpu = HarrisCorner::create();

    // New mat has height/width the same as the old
    Mat image_corners = Mat::zeros(height, width, CV_8U);
    
    cout << "writing output image " << argv[2] << endl;
    imwrite (argv[2], transpose);

    return 0;
    
}
