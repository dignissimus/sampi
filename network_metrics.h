#ifndef NETWORK_METRICS_H
#define NETWORK_METRICS_H

#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

inline long long get_port_xmit_wait() {
    long long total_wait = 0;
    std::string ib_path = "/sys/class/infiniband";
    if (!std::filesystem::exists(ib_path)) {
        return -1;
    }
    try {
        for (const auto& entry : std::filesystem::directory_iterator(ib_path)) {
            std::string counter_path = entry.path().string() + "/ports/1/counters/port_xmit_wait";
            std::ifstream file(counter_path);
            if (file.is_open()) {
                long long val = 0;
                if (file >> val) {
                    total_wait += val;
                }
            }
        }
    } catch (...) {
        return -1;
    }
    return total_wait;
}

inline void print_network_metrics(const char* phase_name) {
    int world_rank, world_size_local;
    PMPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &world_size_local);

    long long wait_start = -1;
    if (world_rank == 0) {
        wait_start = get_port_xmit_wait();
    }

    int num_bytes = 500 * 1024 * 1024;
    std::vector<char> buffer(num_bytes, 0);
    
    PMPI_Bcast(buffer.data(), num_bytes, MPI_CHAR, 0, MPI_COMM_WORLD);
    PMPI_Barrier(MPI_COMM_WORLD);
    
    auto start = std::chrono::high_resolution_clock::now();
    PMPI_Bcast(buffer.data(), num_bytes, MPI_CHAR, 0, MPI_COMM_WORLD);
    PMPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (world_rank == 0) {
        long long wait_end = get_port_xmit_wait();
        double duration = std::chrono::duration<double>(end - start).count();
        double bandwidth = (num_bytes / (1024.0 * 1024.0)) / duration;
        std::cout << "[METRIC] [" << phase_name << "] Cluster Bcast Bandwidth: " << bandwidth << " MB/s (took " << duration << "s)" << std::endl;
        
        if (wait_start >= 0 && wait_end >= 0) {
            std::cout << "[METRIC] [" << phase_name << "] InfiniBand port_xmit_wait delta: " << (wait_end - wait_start) << " ticks" << std::endl;
        } else {
            std::cout << "[METRIC] [" << phase_name << "] InfiniBand port_xmit_wait: Not available on this system" << std::endl;
        }
    }
}

#endif
