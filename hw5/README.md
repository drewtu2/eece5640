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

**Variable Classes:** Demonstrated by the variable classes experiment run with method A. 
Having more classes resulted in long run times. There are two likely explanations 
for this:
    1. With more classes, more comparisions must be made for each term to find 
    the correct bucket to place the value in
    2. With more buckets to place the term in, the memory access pattern becomes 
    evern more complicated than before. 

**Variable Problem Size:** Demonstrated by **8 Node Variable N** test case run with method A.
With a larger N, the problem size grows and thus, we should expect the run time
to grow as well. We see this behavior clearly in the the timing results. Initially,
an increase in problem size by a factor of 10 led to a small increase in run time,
(less tha an equivocal 10x increase in time). This is likely because the underlying
hardware was being fully utilized in the smaller cases, thus increased problem sizes 
took advantage of idle hardware. However, the 10x jumps in problem size after 
100,000 elements resulted in similar 10x jumps in the timing 
data. 

**Method A vs. Method B**:




