# Binning in MPI and Cuda

## Method A

### 4 Node Variable Clases
| Nodes | # Elements | # Classes | Time   |
|-------|------------|-----------|--------|
| 4     | 10,000     | 1         | 70 us  |
| 4     | 10,000     | 5         | 130 us |
| 4     | 10,000     | 10        | 157 us |
| 4     | 10,000     | 20        | 290 us |
| 4     | 10,000     | 40        | 323 us |

### 8 Node Variable Clases
| Nodes | # Elements | # Classes | Time   |
|-------|------------|-----------|--------|
| 8     | 10,000     | 1         | 65 us  |
| 8     | 10,000     | 5         | 90 us  |
| 8     | 10,000     | 10        | 95 us  |
| 8     | 10,000     | 20        | 150 us |
| 8     | 10,000     | 40        | 170 us |

### 8 Node Variable N 
| Nodes | # Elements | # Classes | Time       |
|-------|------------|-----------|------------|
| 8     | 1000       | 5         | 35 us      |
| 8     | 10,000     | 5         | 95 us      |
| 8     | 100,000    | 5         | 571 us     |
| 8     | 1,000,000  | 5         | 5399 us    |
| 8     | 10,000,000 | 5         | 53,570 us  |
| 8     | 100,000,000| 5         | 527,793 us |

## Method B
### 4 Node and 8 Node Variable N and 
| Nodes | # Elements | # Classes | Time   |
|-------|------------|-----------|--------|
| 4     | 10,000     | 5         | 249 us |
| 4     | 100,000    | 5         | 2179 us|
| 8     | 10,000     | 5         | 204 us |
| 8     | 100,000    | 5         | 2167 us|

## Cuda A - Variable N
| # Elements  | # Classes | Total Time (w/ Copy) | Run Time (Kernel Only) |
|-------------|-----------|----------------------|------------------------|
| 10,000      | 5         | 1205 ms              | 252 us                 |
| 100,000     | 5         | 1172 ms              | 300 us                 |
| 1,000,000   | 5         | 1142 ms              | 400 us                 |
| 10,000,000  | 5         | 1144 ms              | 448 us                 |
| 100,000,000 | 5         | 1282 ms              | 651 us                 |

## Analysis
Effect of...

**Variable Classes:** 
![Method A Runtime](method_a_runtime.png)

Demonstrated by the variable classes experiment run with method A. 
Having more classes resulted in long run times. There are two likely explanations 
for this:
    1. With more classes, more comparisions must be made for each term to find 
    the correct bucket to place the value in
    2. With more buckets to place the term in, the memory access pattern becomes 
    evern more complicated than before. 

**Variable Problem Size:** 
![Method A vs CUDA](method_a_vs_cuda.png)
![Method B with variable N](method_b.png)

Demonstrated by the **8 Node Variable N** test case run with method A.
With a larger N, the problem size grows and thus, we should expect the run time
to grow as well. We see this behavior clearly in the the timing results. Initially,
an increase in problem size by a factor of 10 led to a small increase in run time,
(less tha an equivocal 10x increase in time). This is likely because the underlying
hardware was being fully utilized in the smaller cases, thus increased problem sizes 
took advantage of idle hardware. However, the 10x jumps in problem size after 
100,000 elements resulted in similar 10x jumps in the timing 
data. 

**Method A vs. Method B**:
![Method A vs Method B](a_vs_b.png)

One of the most interesting questions to consider is how method A differs 
from the method B in timing. 

In method A, a fraction of the data is read and completely sorted by each process. 
The end results are coalesced on node 0 at the end of the run time. 
In comparision, each node in method B reads the entire data stream and only pulls
out nodes from a specific bucket type.

Our results show that method A is far superior to method B, in the 100k element 
case, running almost 4 times faster. This makes sense if we consider the how the
algorithm is running. In method A, every piece of data is touched only once by
any node before being sorted. Comparitively, in method B, each piece of data is
touched by EVERY SINGLE NODE.

**Method A vs. CUDA**
![Method A vs CUDA](method_a_vs_cuda.png)

Another highly interesting question to consider is how well a CUDA implementation 
of binning performs against the algorithm discussed in method A. Intutively,
the operations performed to "bin" data" is very simple, simply determining where 
each piece of data falls within a set of buckets. Looking at the total run time
of the CUDA implementation, we see somewhat dreadful results, taking over 1s to
run on a problem size of 10k entries, much slower than the sub ms results enjoyed
by the MPI implementation. However, we notice the run time remains fairly constant
from problem size to problem size, while the run time from Method A increases
linearly. Based on our projections, it'd be safe to assume that at a problem size
of the next order of magnitude (1e9), we'd expect to see the CUDA version outperform
the MPI version. 

Looking into the problem further, we can evaluate why the CUDA version performs
so poorly compared to the MPI counterpart. By evaluating the kernel runtime itself,
we find the kernel takes a relatively short time to run - only on the order of 100's 
of us. This result is incredibily important because it shows the majority of the 
the time associated with the CUDA implementation has to do with transferring the
data from the host to the device. There are 3 expensive operations that need to 
take place in order for the kernel to run. 

1. cudaMalloc
2. cudaMemcpy (host to device)
3. cudaMemcpy (device to host)

These memory operations are often expensive and are likely one of the leading
cause of the CUDA high run time implementation. 


