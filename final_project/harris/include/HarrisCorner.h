#ifndef __HARRIS_CORNER_H__
#define __HARRIS_CORNER_H__

#include <opencv2/features2d.hpp>
#include <opencv2/core/mat.hpp>

using cv::InputArray;
using cv::InputArrayOfArrays;
using cv::OutputArray;
using cv::OutputArrayOfArrays;
using cv::KeyPoint;
using cv::noArray;

class HarrisCorner {

 private:
  float k;
  float corner_response_threshold;

  void calculate_gradients(OutputArray ix, OutputArray iy, InputArray input);

 public:

  /**
   * Factor method to create a Harris corner detector object
   */
  static HarrisCorner* create(float k, float threshold);
  
  /**
   * Creates an OpenMP accelerated Harris Corner Detector
   */
  static HarrisCorner* createOpenMP(float k, float threshold);

  /**
   * Default constructor for Harris Corner Detector
   */
  HarrisCorner();

  /**
   * Choose different k constructor for Harris Corner Detector
   */
  HarrisCorner(float k, float threshold);


  /**
   * Populates a vector with a list of corner keypoints detected in the given
   * image. Mask can dictate what parts of the image to use. 
   */
  void detect(InputArray image, std::vector< KeyPoint > &keypoints, InputArray mask=noArray());

  /**
   * Detects Harris Corners across multiple images.
   *
   */
  void detect(InputArrayOfArrays images, std::vector<std::vector< KeyPoint > > &keypoints, InputArrayOfArrays masks=noArray());


  /**
   * Computes Harris Corner Images on a single image 
   * 
   * @param image: the image to detect corners on
   * @param keypoints: the keypoints where the points were found
   * @param descriptors: an array of descriptors for each keypoint
   */
  void compute(InputArray image, std::vector<KeyPoint> &keypoints, OutputArray descriptors);

  /**
   * Computes Harris Corner Images on a series of images
   * 
   * @param image: the images to detect corners on
   * @param keypoints: the keypoints where the points were found a vector for
   * each image
   * @param descriptors: an array of descriptors for each keypoint. an array for
   * each image
   */
  void compute(InputArrayOfArrays images, std::vector<std::vector<KeyPoint>> &keypoints, OutputArrayOfArrays descriptors);

};

#endif
