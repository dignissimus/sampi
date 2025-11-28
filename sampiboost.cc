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
#include "fortran_headers.h"

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

std::map<std::pair<int,int>, double> compute_latency_map() {
    int world_rank, world_size;
    PMPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int PING_PONGS = 50;
    const int TAG = 999;
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
    return latency_map;

}

int MPI_Init(int *argc, char ***argv) {
    std::cout << "[ALARM] MPI_Init called" << std::endl;
    int return_value = PMPI_Init(argc, argv);

    int world_rank, world_size;
    PMPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &world_size);

    global_rank = world_rank;

    latency_map = compute_latency_map();

    auto rank_comm = read_communication_profile("sampi_communication_profile.txt", world_size);
    rank_map = compute_rank_map(world_size, latency_map, rank_comm);
    int new_rank = rank_map[world_rank];
    PMPI_Comm_split(MPI_COMM_WORLD, 0, new_rank, &reorder_comm_world);
    int reorder_rank, reorder_size;
    PMPI_Comm_rank(reorder_comm_world, &reorder_rank);
    PMPI_Comm_size(reorder_comm_world, &reorder_size);
    std::cout << "[ALARM] Re-ordered ranks" << std::endl;
    return return_value;
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided) {
    std::cout << "[ALARM] MPI_Init_thread called" << std::endl;
    return PMPI_Init_thread(argc, argv, required, provided);
}

int MPI_Finalize() {
    std::cout << "[ALARM] MPI_Finalize called" << std::endl;
    if (reorder_comm_world != MPI_COMM_NULL) {
        PMPI_Comm_free(&reorder_comm_world);
    }
    return PMPI_Finalize();
}

int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm) {
    return PMPI_Comm_dup(reorder(comm), newcomm);
}

int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm) {
    return PMPI_Comm_split(reorder(comm), color, key, newcomm);
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

int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag,
                 MPI_Comm comm, MPI_Status *status) {

    return PMPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag,
                         recvbuf, recvcount, recvtype, source, recvtag,
                         reorder(comm), status);
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
    // std::cout << "[HOOK] MPI_Barrier called" << std::endl;
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


int MPI_Cart_create(MPI_Comm comm_old, int ndims, const int dims[], const int periods[], int reorder, MPI_Comm *comm_cart) {
    return PMPI_Cart_create(reorder(comm_old), ndims, dims, periods, reorder, comm_cart);
}


static MPI_Fint reorder_comm_f(MPI_Fint f_comm) {
    MPI_Comm c_comm = MPI_Comm_f2c(f_comm);
    if (c_comm == MPI_COMM_WORLD && reorder_comm_world != MPI_COMM_NULL) {
        return MPI_Comm_c2f(reorder_comm_world);
    }
    return f_comm;
}

// todo: deduplicate initialisation logic
extern "C" {

    void mpi_init_(int *ierror) {
        pmpi_init_(ierror);
        if (*ierror != MPI_SUCCESS) return;

        int world_rank, world_size, ierr;
        MPI_Fint f_comm_world = MPI_Comm_c2f(MPI_COMM_WORLD);

        pmpi_comm_rank_(&f_comm_world, &world_rank, &ierr);
        pmpi_comm_size_(&f_comm_world, &world_size, &ierr);

        global_rank = world_rank;

        latency_map = compute_latency_map();

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

        MPI_Fint f_new_comm;
        int color = 0;
        pmpi_comm_split_(&f_comm_world, &color, &new_rank, &f_new_comm, &ierr);
        reorder_comm_world = MPI_Comm_f2c(f_new_comm);

        int reorder_rank, reorder_size;
        pmpi_comm_rank_(&f_new_comm, &reorder_rank, &ierr);
        pmpi_comm_size_(&f_new_comm, &reorder_size, &ierr);

        std::cout << "[ALARM] MPI_Init called" << std::endl;
    }

    // todo: need to reorder in init thread
    void mpi_init_thread_(int *required, int *provided, int *ierror) {
        std::cout << "[ALARM] MPI_Init_thread called" << std::endl;
        pmpi_init_thread_(required, provided, ierror);
    }

    void mpi_finalize_(int *ierror) {
        std::cout << "[ALARM] MPI_Finalize called" << std::endl;
        pmpi_finalize_(ierror);
        if (reorder_comm_world != MPI_COMM_NULL) {
            MPI_Fint f_reorder_comm = MPI_Comm_c2f(reorder_comm_world);
            int ierr;
            pmpi_comm_free_(&f_reorder_comm, &ierr);
            reorder_comm_world = MPI_COMM_NULL;
        }
    }

    void mpi_send_(void *buf, int *count, MPI_Fint *datatype, int *dest, int *tag, MPI_Fint *comm, int *ierror) {
        MPI_Fint reordered_comm = reorder_comm_f(*comm);
        pmpi_send_(buf, count, datatype, dest, tag, &reordered_comm, ierror);
    }

    void mpi_recv_(void *buf, int *count, MPI_Fint *datatype, int *source, int *tag, MPI_Fint *comm, MPI_Fint *status_f, int *ierror) {
        MPI_Fint reordered_comm = reorder_comm_f(*comm);
        pmpi_recv_(buf, count, datatype, source, tag, &reordered_comm, status_f, ierror);
    }

    void mpi_sendrecv_(
      void *sendbuf, int *sendcount, MPI_Fint *sendtype, int *dest, int *sendtag,
      void *recvbuf, int *recvcount, MPI_Fint *recvtype, int *source, int *recvtag,
      MPI_Fint *comm, MPI_Fint *status_f, int *ierror
    ) {
      MPI_Fint reordered_comm = reorder_comm_f(*comm);
      pmpi_sendrecv_(
        sendbuf,
        sendcount,
        sendtype,
        dest,
        sendtag,
        recvbuf,
        recvcount,
        recvtype,
        source,
        recvtag,
        &reordered_comm,
        status_f,
        ierror
      );
    }

    void mpi_isend_(void *buf, int *count, MPI_Fint *datatype, int *dest, int *tag, MPI_Fint *comm, MPI_Fint *request_f, int *ierror) {
        MPI_Fint reordered_comm = reorder_comm_f(*comm);
        pmpi_isend_(buf, count, datatype, dest, tag, &reordered_comm, request_f, ierror);
    }

    void mpi_irecv_(void *buf, int *count, MPI_Fint *datatype, int *source, int *tag, MPI_Fint *comm, MPI_Fint *request_f, int *ierror) {
        MPI_Fint reordered_comm = reorder_comm_f(*comm);
        pmpi_irecv_(buf, count, datatype, source, tag, &reordered_comm, request_f, ierror);
    }

    void mpi_wait_(MPI_Fint *request_f, MPI_Fint *status_f, int *ierror) {
        pmpi_wait_(request_f, status_f, ierror);
    }

    void mpi_waitall_(int *count, MPI_Fint *requests_f, MPI_Fint *statuses_f, int *ierror) {
        pmpi_waitall_(count, requests_f, statuses_f, ierror);
    }

    void mpi_bcast_(void *buffer, int *count, MPI_Fint *datatype, int *root, MPI_Fint *comm, int *ierror) {
        MPI_Fint reordered_comm = reorder_comm_f(*comm);
        pmpi_bcast_(buffer, count, datatype, root, &reordered_comm, ierror);
    }

    void mpi_reduce_(const void *sendbuf, void *recvbuf, int *count, MPI_Fint *datatype, MPI_Fint *op, int *root, MPI_Fint *comm, int *ierror) {
        MPI_Fint reordered_comm = reorder_comm_f(*comm);
        pmpi_reduce_(sendbuf, recvbuf, count, datatype, op, root, &reordered_comm, ierror);
    }

    void mpi_allreduce_(const void *sendbuf, void *recvbuf, int *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, int *ierror) {
        MPI_Fint reordered_comm = reorder_comm_f(*comm);
        pmpi_allreduce_(sendbuf, recvbuf, count, datatype, op, &reordered_comm, ierror);
    }

    void mpi_scatter_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype, void *recvbuf, int *recvcount, MPI_Fint *recvtype, int *root, MPI_Fint *comm, int *ierror) {
        MPI_Fint reordered_comm = reorder_comm_f(*comm);
        pmpi_scatter_(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, &reordered_comm, ierror);
    }

    void mpi_gather_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype, void *recvbuf, int *recvcount, MPI_Fint *recvtype, int *root, MPI_Fint *comm, int *ierror) {
        MPI_Fint reordered_comm = reorder_comm_f(*comm);
        pmpi_gather_(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, &reordered_comm, ierror);
    }

    void mpi_allgather_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype, void *recvbuf, int *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, int *ierror) {
        MPI_Fint reordered_comm = reorder_comm_f(*comm);
        pmpi_allgather_(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, &reordered_comm, ierror);
    }

    void mpi_alltoall_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype, void *recvbuf, int *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, int *ierror) {
        MPI_Fint reordered_comm = reorder_comm_f(*comm);
        pmpi_alltoall_(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, &reordered_comm, ierror);
    }

    void mpi_barrier_(MPI_Fint *comm, int *ierror) {
        MPI_Fint reordered_comm = reorder_comm_f(*comm);
        pmpi_barrier_(&reordered_comm, ierror);
    }

    void mpi_comm_rank_(MPI_Fint *comm, int *rank, int *ierror) {
        MPI_Fint reordered_comm = reorder_comm_f(*comm);
        pmpi_comm_rank_(&reordered_comm, rank, ierror);
    }

    void mpi_comm_size_(MPI_Fint *comm, int *size, int *ierror) {
        MPI_Fint reordered_comm = reorder_comm_f(*comm);
        pmpi_comm_size_(&reordered_comm, size, ierror);
    }

    void mpi_cart_create_(MPI_Fint *comm_old, int *ndims, int *dims, int *periods, int *reorder_f, MPI_Fint *comm_cart_f, int *ierror) {
        MPI_Fint reordered_comm = reorder_comm_f(*comm_old);
        pmpi_cart_create_(&reordered_comm, ndims, dims, periods, reorder_f, comm_cart_f, ierror);
    }

    void mpi_comm_dup_(MPI_Fint *comm, MPI_Fint *newcomm_f, int *ierror) {
        MPI_Fint reordered_comm_old = reorder_comm_f(*comm);
        pmpi_comm_dup_(&reordered_comm_old, newcomm_f, ierror);
    }

    void mpi_comm_split_(MPI_Fint *comm, int *color, int *key, MPI_Fint *newcomm_f, int *ierror) {
        MPI_Fint reordered_comm_old = reorder_comm_f(*comm);
        pmpi_comm_split_(&reordered_comm_old, color, key, newcomm_f, ierror);
    }

}
