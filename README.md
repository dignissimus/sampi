# sampi

# Usage

```bash
# Produces build/libsampiprofile.so and build/libsampiboost.so
bash build.sh
```

```bash
# Produces sampi_communication_profile.txt
LD_PRELOAD=$(realpath /path/to/sampi/build/libsampiprofile.so) mpirun ...

LD_PRELOAD=$(realpath /path/to/sampi/build/libsampiboost.so) mpirun ...
```

# Benchmark Results

## Graph500


We evaluated Sampi using the MPI Graph500 benchmark in a weak scaling configuration on Durham University's Hamilton cluster using 64 cores per node, The University of Edinburgh's ARCHER2 cluster using 64 cores per node, and again on the ARCHER2 cluster using 128 cores per node.

Broadly, the profiling overhead is minimal and automated rank reordering gives a >40% improvement on 1 and 2 node runs when benchmarked on Durham's hamilton cluster. When using 4 nodes, there is a >10% improvement on 4 node runs.

On the ARCHER2 cluster, profiling incurs up to a 25% overhead and rank-reordering improved run time by over 20% on 1 and 2 node runs.

These were tested at the 99% significance level (p < 0.01).

Detailed results and configurations are available in the [Graph500 Benchmark README](benchmarks/graph500/README.md).

![Speedup Plot](benchmarks/graph500/analysis/images/speedup_plot.png)
