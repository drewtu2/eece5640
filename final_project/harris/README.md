# Proposal: High Performance Harris Corner Detector

## Motivation 
I've decided to switch from exploring Horovod in favor of implementing
the Harris Corner Detector in CUDA to gain more experience in developing in CUDA. 
While Horovod is certainly an interesting project, I feel that gaining a stronger
fundamental knowledge in CUDA is more directly applicable to my current interests. 
For example, my work in my upcoming co-op at Optimus Ride (Fall 2019) on the 
mapping and localizatino team will likely involve more high performant code than
need for distributed deep learning training. 

The Harris Corner Detector is one of the original corner detection methods used
for detecting corners in images. It is a staple in the comptuter vision community
and a problem that will lend itself nicely towards development in CUDA due to the
highly data parallel nature of the program.

## Expected Work and Grades
- A: Implemented Harris Corner sucessfully in CUDA w/ a detailed results report
comparing performance between different versions of the program
- A-: Harris corner in CUDA but with more in depth CUDA functionality. See note below. 
- B+: Implemented Naive Implementation of Harris Corner sucessfully in CUDA, no report
- B: Implemented Harris Corner sucessfully in OpenMP
- B-: Implemented Harris Corner sucessfully, no parallel acceleration 

All work will be completed in C++/CUDA. For the CUDA portion of this assignment,
the goal is to really explore the differnt knobs and levers offerd by CUDA. This
may include exploring different types of parallelism, synchronizations, methods, 
etc. Without knowing more about CUDA it is a little difficult to say what may 
be immediate useful/applicable to this problem. Furthermore, it is difficult to
know whether Harris Corner itself presents a complicated enough problem to take
advantage of all of these knobs. These details will be explored in the project
report. 

## Stretch Work
Time permitting, I would also be interested in expanding this work in one (or more)
of the following ways. This may occur for this class or for a future side project.
- implement multi-scale functionality 
- rewrite as an OpenCV compatible feature detector class
- tiling functions 
