#define _GNU_SOURCE
#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <string>
#include <fstream>
#include <map>

// TODO: Can replace with just a vector
// Since I only record sends from global_rank
static std::map<std::pair<int,int>, unsigned long long int> partial_rank_chatter;
static std::map<std::string, std::vector<int>> communicator_participants;
std::map<std::string, int> split_count;
static int global_rank;
static int global_world_size;

std::string COMM_NAME(const MPI_Comm& comm) {
  char comm_name[MPI_MAX_OBJECT_NAME];
  int comm_name_length;
  MPI_Comm_get_name(comm, comm_name, &comm_name_length);
  return std::string(comm_name);
}

void INC_COMM(const MPI_Comm& comm) {
    for (const int participant : communicator_participants[COMM_NAME(comm)]) {
    ++partial_rank_chatter[{global_rank, participant}];
  }
}

void INC(int source, int dest) {
  ++partial_rank_chatter[{source, dest}];
}

int MAP(const MPI_Comm& comm, int rank) {
  return communicator_participants[COMM_NAME(comm)][rank];
}

int MPI_Init(int *argc, char ***argv) {
    int return_value = PMPI_Init(argc, argv);
    int world_rank, world_size;
    PMPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &world_size);

    char world_rank_name[MPI_MAX_OBJECT_NAME];
    int world_rank_name_length;
    MPI_Comm_get_name(MPI_COMM_WORLD, world_rank_name, &world_rank_name_length);
    for (int i = 0; i < world_size; ++i) {
      communicator_participants[std::string(world_rank_name)].push_back(i); 
    }

    global_rank = world_rank;
    global_world_size = world_size;
    return return_value;
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided) {
    std::cout << "[PROFILE] MPI_Init_thread called (required=" << required << ")" << std::endl;
    return PMPI_Init_thread(argc, argv, required, provided);
}

void write_rank_chatter_to_file(const std::vector<std::vector<unsigned long long int>>& all_rank_chatter) {
    std::ofstream outfile("sampi_communication_profile.txt");
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open output file" << std::endl;
        return;
    }
    
    int world_size = all_rank_chatter.size();
    
    for (const auto& row : all_rank_chatter) {
      for (const auto& number : row) {
        outfile << number << ' ';
      }
      outfile << std::endl;
    }
        
    outfile.close();
    std::cout << "[PROFILE] Communication data written to sampi_communication_profile.txt" << std::endl;
}

int MPI_Finalize() {
    std::cout << "[PROFILE] MPI_Finalize called" << std::endl;
    
    if (global_rank == 0) {
        std::vector<std::vector<unsigned long long int>> all_rank_chatter(
          global_world_size, 
          std::vector<unsigned long long int>(global_world_size, 0)
        );
        
        for (int i = 0; i < global_world_size; ++i) {
            all_rank_chatter[0][i] = partial_rank_chatter[{0, i}];
            all_rank_chatter[i][0] = partial_rank_chatter[{0, i}];
        }
        
        for (int i = 1; i < global_world_size; ++i) {
            std::vector<unsigned long long int> rank_data(global_world_size, 0);
            MPI_Recv(rank_data.data(), global_world_size, MPI_LONG_LONG, i, 0, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < global_world_size; ++j) {
              all_rank_chatter[i][j] += rank_data[j];
              all_rank_chatter[j][i] += rank_data[j];
            }
        }
        
        write_rank_chatter_to_file(all_rank_chatter);
    }
    else {
        std::vector<unsigned long long int> send_to_vector(global_world_size, 0);
        for (int i = 0; i < global_world_size; ++i) {
            send_to_vector[i] = partial_rank_chatter[{global_rank, i}];
        }
        MPI_Send(send_to_vector.data(), global_world_size, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    return PMPI_Finalize();
}

// point to point
int MPI_Send(const void *buf, int count, MPI_Datatype datatype,
             int dest, int tag, MPI_Comm comm) {
    INC(global_rank, MAP(comm, dest));
    std::cout << "[PROFILE] MPI_Send -> dest=" << dest << ", tag=" << tag
              << ", count=" << count << std::endl;
    return PMPI_Send(buf, count, datatype, dest, tag, comm);
}

int MPI_Recv(void *buf, int count, MPI_Datatype datatype,
             int source, int tag, MPI_Comm comm, MPI_Status *status) {
    std::cout << "[PROFILE] MPI_Recv <- source=" << source << ", tag=" << tag
              << ", count=" << count << std::endl;
    return PMPI_Recv(buf, count, datatype, source, tag, comm, status);
}

int MPI_Isend(const void *buf, int count, MPI_Datatype datatype,
              int dest, int tag, MPI_Comm comm, MPI_Request *request) {
    INC(global_rank, MAP(comm, dest));
    std::cout << "[PROFILE] MPI_Isend -> dest=" << dest << ", tag=" << tag << std::endl;
    return PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
}

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
              int source, int tag, MPI_Comm comm, MPI_Request *request) {
    std::cout << "[PROFILE] MPI_Irecv <- source=" << source << ", tag=" << tag << std::endl;
    return PMPI_Irecv(buf, count, datatype, source, tag, comm, request);
}

int MPI_Wait(MPI_Request *request, MPI_Status *status) {
    std::cout << "[PROFILE] MPI_Wait called" << std::endl;
    return PMPI_Wait(request, status);
}

int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]) {
    std::cout << "[PROFILE] MPI_Waitall called (" << count << " requests)" << std::endl;
    return PMPI_Waitall(count, array_of_requests, array_of_statuses);
}

// collective
int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
              int root, MPI_Comm comm) {
    INC_COMM(comm);
    std::cout << "[PROFILE] MPI_Bcast from root=" << root << ", count=" << count << std::endl;
    return PMPI_Bcast(buffer, count, datatype, root, comm);
}

int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
               MPI_Op op, int root, MPI_Comm comm) {
    std::cout << "[PROFILE] MPI_Reduce -> root=" << root << ", count=" << count << std::endl;
    return PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
}

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                  MPI_Op op, MPI_Comm comm) {
    INC_COMM(comm);
    std::cout << "[PROFILE] MPI_Allreduce called (" << count << " elements)" << std::endl;
    return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
}

int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int root, MPI_Comm comm) {
    INC_COMM(comm);
    std::cout << "[PROFILE] MPI_Scatter from root=" << root << std::endl;
    return PMPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
}

int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype,
               int root, MPI_Comm comm) {
    INC_COMM(comm);
    std::cout << "[PROFILE] MPI_Gather to root=" << root << std::endl;
    return PMPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
}

int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  MPI_Comm comm) {

    INC_COMM(comm);
    std::cout << "[PROFILE] MPI_Allgather called (" << sendcount << " elements per rank)" << std::endl;
    return PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
}

int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 MPI_Comm comm) {

    INC_COMM(comm);
    std::cout << "[PROFILE] MPI_Alltoall called" << std::endl;
    return PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
}

int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm) {
  int result = PMPI_Comm_split(comm, color, key, newcomm);

  char original_name[MPI_MAX_OBJECT_NAME];
  int original_name_length;
  MPI_Comm_get_name(comm, original_name, &original_name_length);

  int nsplits = ++split_count[std::string(original_name)];
  std::string new_comm_name = std::string(original_name) + " (split) (id=" + std::to_string(nsplits) + ") (colour=" + std::to_string(color) + ")";
  MPI_Comm_set_name(*newcomm, new_comm_name.c_str());

  int new_size;
  PMPI_Comm_size(*newcomm, &new_size);
  std::vector<int> world_ranks_array(new_size);
  PMPI_Allgather(
    &global_rank,
    1,
    MPI_INT,
    world_ranks_array.data(),
    1,
    MPI_INT,
    *newcomm
  );
  communicator_participants[new_comm_name] = world_ranks_array;
}

int MPI_Cart_create(MPI_Comm comm_old, int ndims, const int dims[], const int periods[], int reorder, MPI_Comm *comm_cart) {
    int result = PMPI_Cart_create(comm_old, ndims, dims, periods, reorder, comm_cart);

    char original_name[MPI_MAX_OBJECT_NAME];
    int original_name_length;
    MPI_Comm_get_name(comm_old, original_name, &original_name_length);

    int nsplits = ++split_count[std::string(original_name)];
    std::string new_comm_name = std::string(original_name) + " (cart) " + "(id=" + std::to_string(nsplits) + ")";
    MPI_Comm_set_name(*comm_cart, new_comm_name.c_str());

    int new_size;
    PMPI_Comm_size(*comm_cart, &new_size);

    std::vector<int> world_ranks_array(new_size);
    PMPI_Allgather(
        &global_rank,
        1,
        MPI_INT,
        world_ranks_array.data(),
        1,
        MPI_INT,
        *comm_cart
    );

    communicator_participants[new_comm_name] = world_ranks_array;
    return result;
}
