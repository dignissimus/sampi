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


We evaluated Sampi using the MPI Graph500 benchmark in a weak scaling configuration on Durham University's Hamilton cluster. Broadly, the profiling overhead is minimal and automated rank reordering gives a >40% improvement on 1 and 2 node runs. When using 4 nodes, there is a >10% improvement on 4 node runs. These were tested at the 99% significance level (p < 0.01).

Detailed results and configurations are available in the [Graph500 Benchmark README](benchmarks/graph500/README.md).

![Speedup Plot](benchmarks/graph500/analysis/images/speedup_plot.png)
