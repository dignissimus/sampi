# sampi

# Usage

```bash
bash build.sh
```

```bash
# Produces sampi_communication_profile.txt
LD_PRELOAD=$(realpath /path/to/sampi/build/libsampiprofile.so) mpirun ...

LD_PRELOAD=$(realpath /path/to/sampi/build/libsampiboost.so) mpirun ...
```
