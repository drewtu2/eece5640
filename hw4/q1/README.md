# Towards the Fastest Implementaiton of Matrix Inversion
1. [Eigen Library](eigen_library)
    A long, long, time ago (~3 years), a NICE professor let me participate as
    a REU student on his project to develop an open source machine learning library.
    As part of that work, we were asked to implement a number of basic linear
    algebra functions, often times, wrapping the funcitons from the Eigen library.
    That project taught us it was a sin to try to reinvent the wheel and not take 
    advantage of some really smart people who have already spent the time 
    optimizing the heck out of the problem we were solving. 

    In this method we use the PartialPivLU method to calcualte the inverse. This
    allows us take advantage of multhreading in the Eigen Library

2. [Parallel Gaussian](parallel_gaussian)
    Of coures... sometimes its the exercise of reinventing the wheel that's 
    important, not necessarily the wheel itself. Performs a gaussian elmination
    which is parallelized by OpenMP. 

[eigen_library]: https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
[parallel_gaussian]: https://www.cs.rutgers.edu/~venugopa/parallel_summer2012/ge.html
[buff_column]:https://cse.buffalo.edu/faculty/miller/Courses/CSE633/thanigachalam-Spring-2014-CSE633.pdf
