#ifndef __PARALLEL_GAUSSIAN_H__
#define __PARALLEL_GAUSSIAN_H__

#include <iostream>
#include <vector>

class ParallelGaussian {
 private:
  int num_row;
  int num_col;
  float** A;
  float** I;

  /**
   * Computes the gaussian_elimination step on a given matrix of size 
   * num_row X num_col. Begins building the inverse of the matrix into I.
   */
  void gaussian_elimination();
  
  /**
   * Computes the back substitution step on a given matrix of size 
   * num_row X num_col. Begins building the inverse of the matrix into I.
   */
  void back_substitution();

  /**
   * Swaps the current row with the next row whose [ii][ii] value does not equal
   * 0
   */
  void row_swap(float** A, int current_row, int num_row);

  /**
   * Get an identity matrix of a given size
   */
  float** get_I(int size);

 public:
  ParallelGaussian(float** A, int num_row, int num_col);
  ~ParallelGaussian();
  float** run();

};
#endif
