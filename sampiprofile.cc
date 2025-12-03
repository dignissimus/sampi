#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mpi.h>
#include <numeric>
#include <string>
#include <vector>
#include "fortran_headers.h"

// TODO: Can replace with just a vector
// Since I only record sends from this process
static std::map<std::pair<int, int>, unsigned long long int>
    partial_rank_communication;
static std::map<std::string, std::vector<int>> communicator_participants;
std::map<std::string, int> split_count;
static int global_rank;
static int global_world_size;

std::string COMM_NAME(const MPI_Comm &comm) {
  char comm_name[MPI_MAX_OBJECT_NAME];
  int comm_name_length;
  MPI_Comm_get_name(comm, comm_name, &comm_name_length);
  return std::string(comm_name);
}

std::string COMM_NAME_F(MPI_Fint f_comm) {
  char comm_name[MPI_MAX_OBJECT_NAME];
  int comm_name_length = 0;
  int ierr = 0;
  pmpi_comm_get_name_(&f_comm, comm_name, &comm_name_length, &ierr,
                      MPI_MAX_OBJECT_NAME);
  if (ierr != MPI_SUCCESS || comm_name_length == 0) {
    std::cerr << "Unable to get name for Fortran communicator with id" << f_comm
              << std::endl;
    return std::string("???");
  }
  return std::string(comm_name, comm_name_length);
}

void INC_COMM(const MPI_Comm &comm) {
  for (const int participant : communicator_participants[COMM_NAME(comm)]) {
    ++partial_rank_communication[{global_rank, participant}];
  }
}

void INC(int source, int dest) { ++partial_rank_communication[{source, dest}]; }

int MAP(const MPI_Comm &comm, int rank) {
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
  return PMPI_Init_thread(argc, argv, required, provided);
}

void write_rank_communication_to_file(
    const std::vector<std::vector<unsigned long long int>> &all_rank_communication) {
  std::ofstream outfile("sampi_communication_profile.txt");
  if (!outfile.is_open()) {
    std::cerr << "Error: Could not open output file sampi_communication_profile.txt" << std::endl;
    return;
  }

  for (const auto &row : all_rank_communication) {
    for (const auto &number : row) {
      outfile << number << ' ';
    }
    outfile << std::endl;
  }

  outfile.close();
  std::cout << "[PROFILE] Communication data written to "
               "sampi_communication_profile.txt"
            << std::endl;
}

int MPI_Finalize() {
  std::cout << "[PROFILE] MPI_Finalize called" << std::endl;

  if (global_rank == 0) {
    std::vector<std::vector<unsigned long long int>> all_rank_communication(
        global_world_size,
        std::vector<unsigned long long int>(global_world_size, 0));

    for (int i = 0; i < global_world_size; ++i) {
      all_rank_communication[0][i] = partial_rank_communication[{0, i}];
      all_rank_communication[i][0] = partial_rank_communication[{0, i}];
    }

    for (int i = 1; i < global_world_size; ++i) {
      std::vector<unsigned long long int> rank_data(global_world_size, 0);
      MPI_Recv(rank_data.data(), global_world_size, MPI_LONG_LONG, i, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      for (int j = 0; j < global_world_size; ++j) {
        all_rank_communication[i][j] += rank_data[j];
        all_rank_communication[j][i] += rank_data[j];
      }
    }

    write_rank_communication_to_file(all_rank_communication);
  } else {
    std::vector<unsigned long long int> send_to_vector(global_world_size, 0);
    for (int i = 0; i < global_world_size; ++i) {
      send_to_vector[i] = partial_rank_communication[{global_rank, i}];
    }
    MPI_Send(send_to_vector.data(), global_world_size, MPI_LONG_LONG, 0, 0,
             MPI_COMM_WORLD);
  }

  PMPI_Barrier(MPI_COMM_WORLD);

  return PMPI_Finalize();
}

// point to point
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
             int tag, MPI_Comm comm) {
  INC(global_rank, MAP(comm, dest));
  return PMPI_Send(buf, count, datatype, dest, tag, comm);
}

int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
             MPI_Comm comm, MPI_Status *status) {
  return PMPI_Recv(buf, count, datatype, source, tag, comm, status);
}

int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 int dest, int sendtag, void *recvbuf, int recvcount,
                 MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm,
                 MPI_Status *status) {

  INC(global_rank, MAP(comm, dest));
  return PMPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf,
                       recvcount, recvtype, source, recvtag, comm, status);
}

int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm, MPI_Request *request) {
  INC(global_rank, MAP(comm, dest));
  return PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
}

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
              MPI_Comm comm, MPI_Request *request) {
  return PMPI_Irecv(buf, count, datatype, source, tag, comm, request);
}

int MPI_Wait(MPI_Request *request, MPI_Status *status) {
  return PMPI_Wait(request, status);
}

int MPI_Waitall(int count, MPI_Request array_of_requests[],
                MPI_Status array_of_statuses[]) {
  return PMPI_Waitall(count, array_of_requests, array_of_statuses);
}

// collective
int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root,
              MPI_Comm comm) {
  INC_COMM(comm);
  return PMPI_Bcast(buffer, count, datatype, root, comm);
}

int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
               MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
  return PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
}

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  INC_COMM(comm);
  return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
}

int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                MPI_Comm comm) {
  INC_COMM(comm);
  return PMPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                      recvtype, root, comm);
}

int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
               MPI_Comm comm) {
  INC_COMM(comm);
  return PMPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                     root, comm);
}

int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  MPI_Comm comm) {

  INC_COMM(comm);
  return PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                        recvtype, comm);
}

int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 MPI_Comm comm) {

  INC_COMM(comm);
  return PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                       recvtype, comm);
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
  PMPI_Allgather(&global_rank, 1, MPI_INT, world_ranks_array.data(), 1, MPI_INT,
                 *newcomm);
  communicator_participants[new_comm_name] = world_ranks_array;
  return result;
}

int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm) {
  int result = PMPI_Comm_dup(comm, newcomm);

  char original_name[MPI_MAX_OBJECT_NAME];
  int original_name_length;
  MPI_Comm_get_name(comm, original_name, &original_name_length);

  int ndups = ++split_count[std::string(original_name)];
  std::string new_comm_name =
      std::string(original_name) + " (dup) (id=" + std::to_string(ndups) + ")";
  MPI_Comm_set_name(*newcomm, new_comm_name.c_str());

  int new_size;
  PMPI_Comm_size(*newcomm, &new_size);
  std::vector<int> world_ranks_array(new_size);
  PMPI_Allgather(&global_rank, 1, MPI_INT, world_ranks_array.data(), 1, MPI_INT,
                 *newcomm);
  communicator_participants[new_comm_name] = world_ranks_array;

  return result;
}

int MPI_Barrier(MPI_Comm comm) {
  INC_COMM(comm);
  return PMPI_Barrier(comm);
}

int MPI_Cart_create(MPI_Comm comm_old, int ndims, const int dims[],
                    const int periods[], int reorder, MPI_Comm *comm_cart) {
  int result =
      PMPI_Cart_create(comm_old, ndims, dims, periods, reorder, comm_cart);

  char original_name[MPI_MAX_OBJECT_NAME];
  int original_name_length;
  MPI_Comm_get_name(comm_old, original_name, &original_name_length);

  int nsplits = ++split_count[std::string(original_name)];
  std::string new_comm_name = std::string(original_name) + " (cart) " +
                              "(id=" + std::to_string(nsplits) + ")";
  MPI_Comm_set_name(*comm_cart, new_comm_name.c_str());

  int new_size;
  PMPI_Comm_size(*comm_cart, &new_size);

  std::vector<int> world_ranks_array(new_size);
  PMPI_Allgather(&global_rank, 1, MPI_INT, world_ranks_array.data(), 1, MPI_INT,
                 *comm_cart);

  communicator_participants[new_comm_name] = world_ranks_array;
  return result;
}

// fortran: todo, deduplicate init and finalise
extern "C" {
void mpi_init_(int *ierror) {
  pmpi_init_(ierror);
  if (*ierror != MPI_SUCCESS)
    return;

  int world_rank, world_size, ierr;
  MPI_Fint f_comm_world = MPI_Comm_c2f(MPI_COMM_WORLD);

  pmpi_comm_rank_(&f_comm_world, &world_rank, &ierr);
  pmpi_comm_size_(&f_comm_world, &world_size, &ierr);

  std::string world_rank_name = COMM_NAME_F(f_comm_world);
  for (int i = 0; i < world_size; ++i) {
    communicator_participants[std::string(world_rank_name)].push_back(i);
  }

  global_rank = world_rank;
  global_world_size = world_size;
}

void mpi_init_thread_(int *required, int *provided, int *ierror) {
  pmpi_init_thread_(required, provided, ierror);
  if (*ierror != MPI_SUCCESS)
    return;

  int world_rank, world_size, ierr;
  MPI_Fint f_comm_world = MPI_Comm_c2f(MPI_COMM_WORLD);

  pmpi_comm_rank_(&f_comm_world, &world_rank, &ierr);
  pmpi_comm_size_(&f_comm_world, &world_size, &ierr);

  char world_rank_name[MPI_MAX_OBJECT_NAME];
  int world_rank_name_length;
  MPI_Comm_get_name(MPI_COMM_WORLD, world_rank_name, &world_rank_name_length);

  for (int i = 0; i < world_size; ++i) {
    communicator_participants[std::string(world_rank_name)].push_back(i);
  }

  global_rank = world_rank;
  global_world_size = world_size;
}

void mpi_finalize_(int *ierror) {
  std::cout << "[PROFILE] MPI_Finalize called" << std::endl;

  MPI_Fint f_comm_world = MPI_Comm_c2f(MPI_COMM_WORLD);
  MPI_Fint f_long_long = MPI_Type_c2f(MPI_LONG_LONG);
  int f_ierr;

  if (global_rank == 0) {
    std::vector<std::vector<unsigned long long int>> all_rank_communication(
        global_world_size,
        std::vector<unsigned long long int>(global_world_size, 0));

    for (int i = 0; i < global_world_size; ++i) {
      all_rank_communication[0][i] = partial_rank_communication[{0, i}];
      all_rank_communication[i][0] = partial_rank_communication[{0, i}];
    }

    for (int i = 1; i < global_world_size; ++i) {
      std::vector<unsigned long long int> rank_data(global_world_size, 0);
      int count = global_world_size;
      int source = i;
      int tag = 0;
      pmpi_recv_(rank_data.data(), &count, &f_long_long, &source, &tag,
                 &f_comm_world, (MPI_Fint *)MPI_F_STATUS_IGNORE, &f_ierr);

      for (int j = 0; j < global_world_size; ++j) {
        all_rank_communication[i][j] += rank_data[j];
        all_rank_communication[j][i] += rank_data[j];
      }
    }
    write_rank_communication_to_file(all_rank_communication);
  } else {
    std::vector<unsigned long long int> send_to_vector(global_world_size, 0);
    for (int i = 0; i < global_world_size; ++i) {
      send_to_vector[i] = partial_rank_communication[{global_rank, i}];
    }
    int count = global_world_size;
    int dest = 0;
    int tag = 0;
    pmpi_send_(send_to_vector.data(), &count, &f_long_long, &dest, &tag,
               &f_comm_world, &f_ierr);
  }

  pmpi_barrier_(&f_comm_world, &f_ierr);

  pmpi_finalize_(ierror);
}

void mpi_send_(void *buf, int *count, MPI_Fint *datatype, int *dest, int *tag,
               MPI_Fint *comm, int *ierror) {
  MPI_Comm c_comm = MPI_Comm_f2c(*comm);
  INC(global_rank, MAP(c_comm, *dest));
  pmpi_send_(buf, count, datatype, dest, tag, comm, ierror);
}

void mpi_recv_(void *buf, int *count, MPI_Fint *datatype, int *source, int *tag,
               MPI_Fint *comm, MPI_Fint *status_f, int *ierror) {
  pmpi_recv_(buf, count, datatype, source, tag, comm, status_f, ierror);
}

void mpi_sendrecv_(void *sendbuf, int *sendcount, MPI_Fint *sendtype, int *dest,
                   int *sendtag, void *recvbuf, int *recvcount,
                   MPI_Fint *recvtype, int *source, int *recvtag,
                   MPI_Fint *comm, MPI_Fint *status_f, int *ierror) {
  MPI_Comm c_comm = MPI_Comm_f2c(*comm);
  INC(global_rank, MAP(c_comm, *dest));
  pmpi_sendrecv_(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf,
                 recvcount, recvtype, source, recvtag, comm, status_f, ierror);
}

void mpi_isend_(void *buf, int *count, MPI_Fint *datatype, int *dest, int *tag,
                MPI_Fint *comm, MPI_Fint *request_f, int *ierror) {
  MPI_Comm c_comm = MPI_Comm_f2c(*comm);
  INC(global_rank, MAP(c_comm, *dest));
  pmpi_isend_(buf, count, datatype, dest, tag, comm, request_f, ierror);
}

void mpi_irecv_(void *buf, int *count, MPI_Fint *datatype, int *source,
                int *tag, MPI_Fint *comm, MPI_Fint *request_f, int *ierror) {
  pmpi_irecv_(buf, count, datatype, source, tag, comm, request_f, ierror);
}

void mpi_wait_(MPI_Fint *request_f, MPI_Fint *status_f, int *ierror) {
  pmpi_wait_(request_f, status_f, ierror);
}

void mpi_waitall_(int *count, MPI_Fint *requests_f, MPI_Fint *statuses_f,
                  int *ierror) {
  pmpi_waitall_(count, requests_f, statuses_f, ierror);
}

void mpi_bcast_(void *buffer, int *count, MPI_Fint *datatype, int *root,
                MPI_Fint *comm, int *ierror) {
  MPI_Comm c_comm = MPI_Comm_f2c(*comm);
  INC_COMM(c_comm);
  pmpi_bcast_(buffer, count, datatype, root, comm, ierror);
}

void mpi_reduce_(const void *sendbuf, void *recvbuf, int *count,
                 MPI_Fint *datatype, MPI_Fint *op, int *root, MPI_Fint *comm,
                 int *ierror) {
  pmpi_reduce_(sendbuf, recvbuf, count, datatype, op, root, comm, ierror);
}

void mpi_allreduce_(const void *sendbuf, void *recvbuf, int *count,
                    MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm,
                    int *ierror) {
  MPI_Comm c_comm = MPI_Comm_f2c(*comm);
  INC_COMM(c_comm);
  pmpi_allreduce_(sendbuf, recvbuf, count, datatype, op, comm, ierror);
}

void mpi_scatter_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype,
                  void *recvbuf, int *recvcount, MPI_Fint *recvtype, int *root,
                  MPI_Fint *comm, int *ierror) {
  MPI_Comm c_comm = MPI_Comm_f2c(*comm);
  INC_COMM(c_comm);
  pmpi_scatter_(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                root, comm, ierror);
}

void mpi_gather_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype,
                 void *recvbuf, int *recvcount, MPI_Fint *recvtype, int *root,
                 MPI_Fint *comm, int *ierror) {
  MPI_Comm c_comm = MPI_Comm_f2c(*comm);
  INC_COMM(c_comm);
  pmpi_gather_(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
               comm, ierror);
}

void mpi_allgather_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype,
                    void *recvbuf, int *recvcount, MPI_Fint *recvtype,
                    MPI_Fint *comm, int *ierror) {
  MPI_Comm c_comm = MPI_Comm_f2c(*comm);
  INC_COMM(c_comm);
  pmpi_allgather_(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                  comm, ierror);
}

void mpi_alltoall_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype,
                   void *recvbuf, int *recvcount, MPI_Fint *recvtype,
                   MPI_Fint *comm, int *ierror) {
  MPI_Comm c_comm = MPI_Comm_f2c(*comm);
  INC_COMM(c_comm);
  pmpi_alltoall_(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                 comm, ierror);
}

void mpi_comm_split_(MPI_Fint *comm, int *color, int *key, MPI_Fint *newcomm_f,
                     int *ierror) {
  pmpi_comm_split_(comm, color, key, newcomm_f, ierror);
  if (*ierror != MPI_SUCCESS)
    return;

  std::string original_name = COMM_NAME_F(*comm);
  int nsplits = ++split_count[std::string(original_name)];
  std::string new_comm_name = std::string(original_name) +
                              " (split) (id=" + std::to_string(nsplits) +
                              ") (colour=" + std::to_string(*color) + ")";
  pmpi_comm_set_name_(newcomm_f, const_cast<char *>(new_comm_name.c_str()),
                      ierror, new_comm_name.length());
  if (*ierror != MPI_SUCCESS)
    return;

  int new_size;
  MPI_Fint f_newcomm = *newcomm_f;
  pmpi_comm_size_(&f_newcomm, &new_size, ierror);
  if (*ierror != MPI_SUCCESS)
    return;

  std::vector<int> world_ranks_array(new_size);
  int sendcount = 1, recvcount = 1;
  MPI_Fint f_int_type = MPI_Type_c2f(MPI_INT);

  pmpi_allgather_(&global_rank, &sendcount, &f_int_type,
                  world_ranks_array.data(), &recvcount, &f_int_type, &f_newcomm,
                  ierror);
  if (*ierror != MPI_SUCCESS)
    return;

  communicator_participants[new_comm_name] = world_ranks_array;
}

void mpi_cart_create_(MPI_Fint *comm_old, int *ndims, int *dims, int *periods,
                      int *reorder, MPI_Fint *comm_cart_f, int *ierror) {
  pmpi_cart_create_(comm_old, ndims, dims, periods, reorder, comm_cart_f,
                    ierror);
  if (*ierror != MPI_SUCCESS)
    return;

  std::string original_name = COMM_NAME_F(*comm_old);
  int nsplits = ++split_count[std::string(original_name)];
  std::string new_comm_name = std::string(original_name) + " (cart) " +
                              "(id=" + std::to_string(nsplits) + ")";
  pmpi_comm_set_name_(comm_cart_f, const_cast<char *>(new_comm_name.c_str()),
                      ierror, new_comm_name.length());
  if (*ierror != MPI_SUCCESS)
    return;

  int new_size;
  MPI_Fint f_cart_comm = *comm_cart_f;
  pmpi_comm_size_(&f_cart_comm, &new_size, ierror);
  if (*ierror != MPI_SUCCESS)
    return;

  std::vector<int> world_ranks_array(new_size);
  int sendcount = 1, recvcount = 1;
  MPI_Fint f_int_type = MPI_Type_c2f(MPI_INT);

  pmpi_allgather_(&global_rank, &sendcount, &f_int_type,
                  world_ranks_array.data(), &recvcount, &f_int_type,
                  &f_cart_comm, ierror);
  if (*ierror != MPI_SUCCESS)
    return;

  communicator_participants[new_comm_name] = world_ranks_array;
}

void mpi_comm_dup_(MPI_Fint *comm, MPI_Fint *newcomm_f, int *ierror) {
  pmpi_comm_dup_(comm, newcomm_f, ierror);

  std::string original_name = COMM_NAME_F(*comm);
  int ndups = ++split_count[std::string(original_name)];
  std::string new_comm_name =
      std::string(original_name) + " (dup) (id=" + std::to_string(ndups) + ")";
  pmpi_comm_set_name_(newcomm_f, const_cast<char *>(new_comm_name.c_str()),
                      ierror, new_comm_name.length());
  if (*ierror != MPI_SUCCESS)
    return;
  int new_size;

  MPI_Fint f_newcomm = *newcomm_f;
  pmpi_comm_size_(&f_newcomm, &new_size, ierror);
  if (*ierror != MPI_SUCCESS)
    return;

  std::vector<int> world_ranks_array(new_size);
  int sendcount = 1, recvcount = 1;
  MPI_Fint f_int_type = MPI_Type_c2f(MPI_INT);

  pmpi_allgather_(&global_rank, &sendcount, &f_int_type,
                  world_ranks_array.data(), &recvcount, &f_int_type, &f_newcomm,
                  ierror);
  if (*ierror != MPI_SUCCESS)
    return;

  communicator_participants[new_comm_name] = world_ranks_array;
}

void mpi_barrier_(MPI_Fint *comm, int *ierror) {
  MPI_Comm c_comm = MPI_Comm_f2c(*comm);
  INC_COMM(c_comm);
  pmpi_barrier_(comm, ierror);
}
}
