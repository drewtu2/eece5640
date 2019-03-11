# Fast Matrix Inversion
- strong scaling:   `t1 / ( N * tN ) * 100%`
- weak scaling:     `( t1 / tN ) * 100%`
## Eigen
|                  | Time        | Strong Scaling | Weak Scaling |
| ---------------- | ----------- | -------------- | ------------ |
| Dense Serial     | 136 ms      | N/A            | N/A          |
| Dense 24 thread  | 138 ms      | 4.1            | 98.0         |
| Sparse Serial    | Time        | Strong Scaling | Weak Scaling |
| Sparse 24 thread | Time        | Strong Scaling | Weak Scaling |

## Gaussian Elimination
|                  | Time        | Strong Scaling | Weak Scaling |
| ---------------- | ----------- | -------------- | ------------ |
| Dense Serial     | 1821 ms     | N/A            | N/A          |
| Dense 24 thread  | 281 ms      | 27.0           | 648.0        |
| Sparse Serial    | Time        | Strong Scaling | Weak Scaling |
| Sparse 24 thread | Time        | Strong Scaling | Weak Scaling |

**Best Configuration for 1024x1024: Eigen w/ 1 thread**


# L1 Cache Size Estimation

# Monte Carlo Estimations of PI
Time to Calculate PI using 1e6 Throws
| Num Processes | Time        |
| ------------- | ----------- |
| 1             |   64 ms     |
| 2             |   47 ms     |
| 4             |   23 ms     |
| 8             |   15 ms     |
| 10            |   10 ms     |

# MPI on Millions of Cores
Developments to positively impact ability to fully exploit parallelism
- Faster means of communication (improved interconnects) reduces propagation
delay of information 
- Improved communication protocols that improve end to end latency to reduce 
overhead of communication
- Larger cache sizes - allows more data to be stored and rapidly accessed
to reduce the likelhood of a cache miss resulting in delays
- Better "many core architectures" and corresponding programming models 
allows for a more efficient use of cores (minimize overhead parallel programming)

# Exascale computing (EC)


