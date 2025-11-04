#define _GNU_SOURCE
#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <map>
#include <utility>

static MPI_Comm reorder_comm_world = MPI_COMM_NULL;
static std::vector<int> rank_map;
static std::map<std::pair<int,int>, double> latency_map;
static int global_rank;
#define reorder(comm) (comm == MPI_COMM_WORLD ? reorder_comm_world : comm)

static double compute_cost(
    const std::vector<int>& mapping,
    const std::map<std::pair<int,int>, double>& latency_map,
    const std::map<std::pair<int,int>, int>& rank_comm,
    int size)
{
    double cost = 0.0;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (i == j) continue;
            int pi = mapping[i];
            int pj = mapping[j];
            auto comm_it = rank_comm.find({pi, pj});
            if (comm_it != rank_comm.end()) {
                cost += comm_it->second * latency_map.at({i, j});
            }
        }
    }
    return cost;
}

std::vector<int> compute_rank_map(
    int size,
    const std::map<std::pair<int,int>, double>& latency_map,
    const std::map<std::pair<int,int>, int>& rank_comm)
{
    std::vector<int> mapping(size);
    std::iota(mapping.begin(), mapping.end(), 0);

    double best_cost = compute_cost(mapping, latency_map, rank_comm, size);
    bool improved = true;

    // 2 opt
    while (improved) {
        improved = false;
        double current_best = best_cost;

        for (int i = 0; i < size; ++i) {
            for (int j = i + 1; j < size; ++j) {
                std::swap(mapping[i], mapping[j]);
                double new_cost = compute_cost(mapping, latency_map, rank_comm, size);

                if (new_cost < best_cost) {
                    best_cost = new_cost;
                    improved = true;
                } else {
                    std::swap(mapping[i], mapping[j]);
                }
            }
        }

        if (!improved) break;
    }

    std::cout << "[HOOK] Runtime rank -> Profile rank";
    for (int i = 0; i < size; ++i)
        std::cout << "Runtime " << i << " -> Profile " << mapping[i] << "\n";

    return mapping;
}

std::map<std::pair<int, int>, int> read_communication_profile(const std::string& filename, int world_size) {
    std::map<std::pair<int, int>, int> rank_comm;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return rank_comm;
    }
    
    std::string line;
    int i = 0;
    while (std::getline(file, line) && i < world_size) {
        std::istringstream iss(line);
        int value;
        int j = 0;
        
        while (iss >> value && j < world_size) {
            rank_comm[{i, j}] = value;
            j++;
        }
        i++;
    }
    file.close();
    
    return rank_comm;
}

int MPI_Init(int *argc, char ***argv) {
    int return_value = PMPI_Init(argc, argv);

    int world_rank, world_size;
    PMPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &world_size);

    global_rank = world_rank;

    const int PING_PONGS = 50;
    const int TAG = 999;

    // 8 bytes = 64 bits
    const int MSG_SIZE = 8;
    std::vector<char> message(MSG_SIZE, 'a');

    for (int i = 0; i < world_size; ++i) {
        for (int j = i+1; j < world_size; ++j) {

            double latency = 0.0;

            PMPI_Barrier(MPI_COMM_WORLD);
            if (world_rank == i || world_rank == j) {
                int other = (world_rank == i) ? j : i;


                auto start = std::chrono::high_resolution_clock::now();

                for (int k = 0; k < PING_PONGS; ++k) {
                    if (world_rank < other) {
                        PMPI_Send(message.data(), MSG_SIZE, MPI_CHAR, other, TAG, MPI_COMM_WORLD);
                        PMPI_Recv(message.data(), MSG_SIZE, MPI_CHAR, other, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    } else {
                        PMPI_Recv(message.data(), MSG_SIZE, MPI_CHAR, other, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        PMPI_Send(message.data(), MSG_SIZE, MPI_CHAR, other, TAG, MPI_COMM_WORLD);
                    }
                }

                auto end = std::chrono::high_resolution_clock::now();
                latency = std::chrono::duration<double>(end - start).count() / (2.0 * PING_PONGS);
            }

            PMPI_Bcast(&latency, 1, MPI_DOUBLE, i, MPI_COMM_WORLD);

            latency_map[{i,j}] = latency;
            latency_map[{j,i}] = latency;
        }
    }

    if (world_rank == 0) {
        std::cout << "[HOOK] Latency matrix:\n";
        for (int i = 0; i < world_size; ++i) {
            for (int j = 0; j < world_size; ++j) {
                if (i == j) std::cout << "0.0 ";
                else std::cout << latency_map[{i,j}] << " ";
            }
            std::cout << "\n";
        }
    }

    auto rank_comm = read_communication_profile("sampi_communication_profile.txt", world_size);
    rank_map = compute_rank_map(world_size, latency_map, rank_comm);
    int new_rank = rank_map[world_rank];
    MPI_Comm_split(MPI_COMM_WORLD, 0, new_rank, &reorder_comm_world);
    int reorder_rank, reorder_size;
    PMPI_Comm_rank(reorder_comm_world, &reorder_rank);
    PMPI_Comm_size(reorder_comm_world, &reorder_size);
    std::cout << "[HOOK] MPI_Init called" << std::endl;
    return return_value;
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided) {
    std::cout << "[HOOK] MPI_Init_thread called (required=" << required << ")" << std::endl;
    return PMPI_Init_thread(argc, argv, required, provided);
}

int MPI_Finalize() {
    std::cout << "[HOOK] MPI_Finalize called" << std::endl;
    return PMPI_Finalize();
}

// point to point
int MPI_Send(const void *buf, int count, MPI_Datatype datatype,
             int dest, int tag, MPI_Comm comm) {
    /*std::cout << "[HOOK] MPI_Send -> dest=" << dest << ", tag=" << tag
              << ", count=" << count << std::endl;*/
    return PMPI_Send(buf, count, datatype, dest, tag, reorder(comm));
}

int MPI_Recv(void *buf, int count, MPI_Datatype datatype,
             int source, int tag, MPI_Comm comm, MPI_Status *status) {
    /*std::cout << "[HOOK] MPI_Recv <- source=" << source << ", tag=" << tag
              << ", count=" << count << std::endl;
*/
    return PMPI_Recv(buf, count, datatype, source, tag, reorder(comm), status);
}

int MPI_Isend(const void *buf, int count, MPI_Datatype datatype,
              int dest, int tag, MPI_Comm comm, MPI_Request *request) {
    // std::cout << "[HOOK] MPI_Isend -> dest=" << dest << ", tag=" << tag << std::endl;
    return PMPI_Isend(buf, count, datatype, dest, tag, reorder(comm), request);
}

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
              int source, int tag, MPI_Comm comm, MPI_Request *request) {
    // std::cout << "[HOOK] MPI_Irecv <- source=" << source << ", tag=" << tag << std::endl;
    return PMPI_Irecv(buf, count, datatype, source, tag, reorder(comm), request);
}

int MPI_Wait(MPI_Request *request, MPI_Status *status) {
    // std::cout << "[HOOK] MPI_Wait called" << std::endl;
    return PMPI_Wait(request, status);
}

int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]) {
    // std::cout << "[HOOK] MPI_Waitall called (" << count << " requests)" << std::endl;
    return PMPI_Waitall(count, array_of_requests, array_of_statuses);
}

// collective
int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
              int root, MPI_Comm comm) {
    // std::cout << "[HOOK] MPI_Bcast from root=" << root << ", count=" << count << std::endl;
    return PMPI_Bcast(buffer, count, datatype, root, reorder(comm));
}

int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
               MPI_Op op, int root, MPI_Comm comm) {
    // std::cout << "[HOOK] MPI_Reduce -> root=" << root << ", count=" << count << std::endl;
    return PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, reorder(comm));
}

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                  MPI_Op op, MPI_Comm comm) {
    // std::cout << "[HOOK] MPI_Allreduce called (" << count << " elements)" << std::endl;
    return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, reorder(comm));
}

int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int root, MPI_Comm comm) {
    // std::cout << "[HOOK] MPI_Scatter from root=" << root << std::endl;
    return PMPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, reorder(comm));
}

int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype,
               int root, MPI_Comm comm) {
    // std::cout << "[HOOK] MPI_Gather to root=" << root << std::endl;
    return PMPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, reorder(comm));
}

int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  MPI_Comm comm) {
    // std::cout << "[HOOK] MPI_Allgather called (" << sendcount << " elements per rank)" << std::endl;
    return PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, reorder(comm));
}

int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 MPI_Comm comm) {
    // std::cout << "[HOOK] MPI_Alltoall called" << std::endl;
    return PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, reorder(comm));
}

int MPI_Barrier(MPI_Comm comm) {
    std::cout << "[HOOK] MPI_Barrier called" << std::endl;
    return PMPI_Barrier(reorder(comm));
}

int MPI_Comm_rank(MPI_Comm comm, int *rank) {
    int ret = PMPI_Comm_rank(reorder(comm), rank);
    // std::cout << "[HOOK] MPI_Comm_rank -> " << *rank << std::endl;
    return ret;
}

int MPI_Comm_size(MPI_Comm comm, int *size) {
    int ret = PMPI_Comm_size(reorder(comm), size);
    // std::cout << "[HOOK] MPI_Comm_size -> " << *size << std::endl;
    return ret;
}


