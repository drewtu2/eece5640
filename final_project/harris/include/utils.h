#ifndef __UTILS_H__
#define __UTILS_H__

#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>

using std::string;
using cv::Mat;
using cv::InputArray;

/**
 * Takes a type (int) from an OpenCV Mat object and prints out the cooresponding 
 * matrix type. 
 * Usefull debugging function taken from StackOverflow (of course):
 * https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv
 */
string type2str(int type);

/**
 * Prints the upper and lower bound of a OpenCV InputArray
 */
void print_bounds(InputArray input, string name);

#endif
