# Graph500 Benchmark

On the Hamilton cluster, profiling incurs a <1% overhead and rank-reordering improved run time by over 40% on 1 and 2 node runs.
On the ARCHER2 cluster, profiling incurs a 25% overhead and rank-reordering improved run time by over 20% on 1 and 2 node runs.

These were tested with a two-sided t-tests at the 99% significance level.

This benchmark suite runs the MPI Graph500 benchmark, testing Breadth-First Search and Single-Source Shortest Path on generated graph dataset. It is configured as a weak scaling experiment where the problem size increases with the compute resources. We ran the benchmarks on Durham's Hamilton cluster using 64 cores per node.

![Speedup Plot](analysis/images/speedup_plot.png)

# Rank Reordering Performance

## Hamilton

### 1 Node

| Algorithm | Mode | Speedup | 99% CI | t-stat | p-value | Cohen's d |
| --------- | ---- | ------- | ------ | ------ | ------- | --------- |
| BFS | Automated Rank Reordering | **+62.7%** | [51.8%, 73.5%] | 15.11 | 4.49e-30 | 2.671 |
| SSSP | Automated Rank Reordering | **+53.8%** | [49.2%, 58.4%] | 30.63 | 3.15e-60 | 5.415 |

### 2 Nodes

| Algorithm | Mode | Speedup | 99% CI | t-stat | p-value | Cohen's d |
| --------- | ---- | ------- | ------ | ------ | ------- | --------- |
| BFS | Automated Rank Reordering | **+44.9%** | [37.6%, 52.1%] | 16.11 | 2.12e-32 | 2.849 |
| SSSP | Automated Rank Reordering | **+42.7%** | [36.5%, 48.9%] | 18.01 | 1.15e-36 | 3.184 |

### 4 Nodes

| Algorithm | Mode | Speedup | 99% CI | t-stat | p-value | Cohen's d |
| --------- | ---- | ------- | ------ | ------ | ------- | --------- |
| BFS | Automated Rank Reordering | **+12.3%** | [7.1%, 17.6%] | 6.13 | 1.04e-08 | 1.084 |
| SSSP | Automated Rank Reordering | **+14.8%** | [10.5%, 19.2%] | 8.95 | 3.81e-15 | 1.583 |

## Archer2, 64 cores per node

### 1 Node

| Algorithm | Mode | Speedup | 99% CI | t-stat | p-value | Cohen's d |
| --------- | ---- | ------- | ------ | ------ | ------- | --------- |
| BFS | Automated Rank Reordering | **+0.8%** | [-0.9%, 2.6%] | 1.25 | 2.14e-01 | 0.221 |
| SSSP | Automated Rank Reordering | **+13.8%** | [9.4%, 18.2%] | 8.27 | 1.65e-13 | 1.461 |

## Archer2, 128 cores per node

### 1 Node

| Algorithm | Mode | Speedup | 99% CI | t-stat | p-value | Cohen's d |
| --------- | ---- | ------- | ------ | ------ | ------- | --------- |
| BFS | Automated Rank Reordering | **+7.2%** | [5.0%, 9.3%] | 8.65 | 1.99e-14 | 1.530 |
| SSSP | Automated Rank Reordering | **+20.1%** | [16.0%, 24.3%] | 12.66 | 3.29e-24 | 2.238 |

### 2 Nodes

| Algorithm | Mode | Speedup | 99% CI | t-stat | p-value | Cohen's d |
| --------- | ---- | ------- | ------ | ------ | ------- | --------- |
| BFS | Automated Rank Reordering | -3.8% | [-6.0%, -1.6%] | -4.61 | 9.58e-06 | -0.816 |
| SSSP | Automated Rank Reordering | **+27.5%** | [23.4%, 31.5%] | 17.64 | 7.64e-36 | 3.119 |

# Profiling Overhead

## Hamilton

### 1 Node

| Algorithm | Mode | Speedup | 99% CI | t-stat | p-value | Cohen's d |
| --------- | ---- | ------- | ------ | ------ | ------- | --------- |
| BFS | Profile Run | +0.0% | [-8.5%, 8.6%] | 0.01 | 9.96e-01 | 0.001 |
| SSSP | Profile Run | +1.2% | [-2.1%, 4.6%] | 0.95 | 3.43e-01 | 0.168 |

### 2 Nodes

| Algorithm | Mode | Speedup | 99% CI | t-stat | p-value | Cohen's d |
| --------- | ---- | ------- | ------ | ------ | ------- | --------- |
| BFS | Profile Run | +0.3% | [-6.0%, 6.5%] | 0.11 | 9.13e-01 | 0.019 |
| SSSP | Profile Run | -0.5% | [-6.1%, 5.0%] | -0.26 | 7.98e-01 | -0.045 |

### 4 Nodes

| Algorithm | Mode | Speedup | 99% CI | t-stat | p-value | Cohen's d |
| --------- | ---- | ------- | ------ | ------ | ------- | --------- |
| BFS | Profile Run | +1.5% | [-4.2%, 7.2%] | 0.69 | 4.91e-01 | 0.122 |
| SSSP | Profile Run | -0.7% | [-5.4%, 4.0%] | -0.40 | 6.90e-01 | -0.071 |

## Archer2, 64 cores per node

### 1 Node

| Algorithm | Mode | Speedup | 99% CI | t-stat | p-value | Cohen's d |
| --------- | ---- | ------- | ------ | ------ | ------- | --------- |
| BFS | Profile Run | -0.6% | [-2.2%, 1.0%] | -0.97 | 3.32e-01 | -0.172 |
| SSSP | Profile Run | -11.8% | [-15.7%, -8.0%] | -8.01 | 6.73e-13 | -1.415 |

## Archer2, 128 cores per node

### 1 Node

| Algorithm | Mode | Speedup | 99% CI | t-stat | p-value | Cohen's d |
| --------- | ---- | ------- | ------ | ------ | ------- | --------- |
| BFS | Profile Run | -2.9% | [-5.1%, -0.8%] | -3.54 | 5.62e-04 | -0.626 |
| SSSP | Profile Run | -18.0% | [-21.4%, -14.5%] | -13.74 | 8.23e-27 | -2.428 |

### 2 Nodes

| Algorithm | Mode | Speedup | 99% CI | t-stat | p-value | Cohen's d |
| --------- | ---- | ------- | ------ | ------ | ------- | --------- |
| BFS | Profile Run | -5.7% | [-7.6%, -3.9%] | -8.16 | 2.96e-13 | -1.442 |
| SSSP | Profile Run | -25.5% | [-28.6%, -22.4%] | -21.37 | 1.03e-43 | -3.777 |
