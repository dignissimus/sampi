#include "fortran_headers.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mpi.h>
#include <numeric>
#include <optional>
#include <string>
#include <vector>
#include "communicators.h"
#include "profiler.h"

int MPI_Init(int *argc, char ***argv) {
  int return_value = PMPI_Init(argc, argv);
  if (return_value != MPI_SUCCESS)
    return return_value;

  int world_rank, world_size;
  PMPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &world_size);

  char world_name[MPI_MAX_OBJECT_NAME];
  int name_len;
  PMPI_Comm_get_name(MPI_COMM_WORLD, world_name, &name_len);

  Profiler::instance().initialise(world_rank, world_size,
                                       std::string(world_name, name_len));

  return return_value;
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided) {
  int result = PMPI_Init_thread(argc, argv, required, provided);
  if (result == MPI_SUCCESS) {
    int rank, size;
    PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &size);

    Profiler::instance().initialise(rank, size,
                                         get_communicator_name(MPI_COMM_WORLD));
  }
  return result;
}

int MPI_Finalize() {
  Profiler::instance().dump_profile();
  PMPI_Barrier(MPI_COMM_WORLD);
  return PMPI_Finalize();
}

int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm) {
  int result = PMPI_Comm_split(comm, color, key, newcomm);

  if (result == MPI_SUCCESS && *newcomm != MPI_COMM_NULL) {
    std::string old_name = get_communicator_name(comm);

    int new_size;
    PMPI_Comm_size(*newcomm, &new_size);
    std::vector<int> world_ranks_array(new_size);

    int dynamic_rank = Profiler::instance().get_global_rank().value_or(0);
    PMPI_Allgather(&dynamic_rank, 1, MPI_INT, world_ranks_array.data(), 1,
                   MPI_INT, *newcomm);

    std::string new_name =
        Profiler::instance().generate_and_track_communicator(
            old_name, "split", color, world_ranks_array);

    PMPI_Comm_set_name(*newcomm, new_name.c_str());
  }
  return result;
}

int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm) {
  int result = PMPI_Comm_dup(comm, newcomm);

  if (result == MPI_SUCCESS && *newcomm != MPI_COMM_NULL) {
    std::string old_name = get_communicator_name(comm);

    int new_size;
    PMPI_Comm_size(*newcomm, &new_size);
    std::vector<int> world_ranks_array(new_size);

    int dynamic_rank = Profiler::instance().get_global_rank().value_or(0);
    PMPI_Allgather(&dynamic_rank, 1, MPI_INT, world_ranks_array.data(), 1,
                   MPI_INT, *newcomm);

    std::string new_name =
        Profiler::instance().generate_and_track_communicator(
            old_name, "dup", 0, world_ranks_array);
    PMPI_Comm_set_name(*newcomm, new_name.c_str());
  }

  return result;
}

// point to point
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
             int tag, MPI_Comm comm) {
  Profiler::instance().record_p2p_send(dest, get_communicator_name(comm));
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
  Profiler::instance().record_p2p_send(dest, get_communicator_name(comm));
  return PMPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf,
                       recvcount, recvtype, source, recvtag, comm, status);
}

int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm, MPI_Request *request) {
  Profiler::instance().record_p2p_send(dest, get_communicator_name(comm));
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

// collectives
int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root,
              MPI_Comm comm) {
  Profiler::instance().record_collective_broadcast(
      get_communicator_name(comm));
  return PMPI_Bcast(buffer, count, datatype, root, comm);
}

int MPI_Scatterv(const void *sendbuf, const int sendcounts[],
                 const int displs[], MPI_Datatype sendtype, void *recvbuf,
                 int recvcount, MPI_Datatype recvtype, int root,
                 MPI_Comm comm) {
  Profiler::instance().record_collective_broadcast(
      get_communicator_name(comm));
  return PMPI_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf,
                       recvcount, recvtype, root, comm);
}

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  Profiler::instance().record_collective_broadcast(
      get_communicator_name(comm));
  return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
}

int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                MPI_Comm comm) {
  Profiler::instance().record_collective_broadcast(
      get_communicator_name(comm));
  return PMPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                      recvtype, root, comm);
}

int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
               MPI_Comm comm) {
  Profiler::instance().record_collective_broadcast(
      get_communicator_name(comm));
  return PMPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                     root, comm);
}

int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  MPI_Comm comm) {
  Profiler::instance().record_collective_broadcast(
      get_communicator_name(comm));
  return PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                        recvtype, comm);
}

int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 MPI_Comm comm) {
  Profiler::instance().record_collective_broadcast(
      get_communicator_name(comm));
  return PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                       recvtype, comm);
}

int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, const int recvcounts[], const int displs[],
                MPI_Datatype recvtype, int root, MPI_Comm comm) {
  Profiler::instance().record_collective_broadcast(
      get_communicator_name(comm));
  return PMPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs,
                      recvtype, root, comm);
}

int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, const int recvcounts[], const int displs[],
                   MPI_Datatype recvtype, MPI_Comm comm) {
  Profiler::instance().record_collective_broadcast(
      get_communicator_name(comm));
  return PMPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts,
                         displs, recvtype, comm);
}

int MPI_Alltoallv(const void *sendbuf, const int sendcounts[],
                  const int sdispls[], MPI_Datatype sendtype, void *recvbuf,
                  const int recvcounts[], const int rdispls[],
                  MPI_Datatype recvtype, MPI_Comm comm) {
  Profiler::instance().record_collective_broadcast(
      get_communicator_name(comm));
  return PMPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf,
                        recvcounts, rdispls, recvtype, comm);
}

int MPI_Barrier(MPI_Comm comm) {
  Profiler::instance().record_collective_broadcast(
      get_communicator_name(comm));
  return PMPI_Barrier(comm);
}

int MPI_Cart_create(MPI_Comm comm_old, int ndims, const int dims[],
                    const int periods[], int reorder, MPI_Comm *comm_cart) {
  int result =
      PMPI_Cart_create(comm_old, ndims, dims, periods, reorder, comm_cart);

  if (result == MPI_SUCCESS && *comm_cart != MPI_COMM_NULL) {
    std::string old_name = get_communicator_name(comm_old);

    int new_size;
    PMPI_Comm_size(*comm_cart, &new_size);
    std::vector<int> world_ranks_array(new_size);

    int dynamic_rank = Profiler::instance().get_global_rank().value_or(0);
    PMPI_Allgather(&dynamic_rank, 1, MPI_INT, world_ranks_array.data(), 1,
                   MPI_INT, *comm_cart);

    std::string new_name =
        Profiler::instance().generate_and_track_communicator(
            old_name, "cart", 0, world_ranks_array);

    PMPI_Comm_set_name(*comm_cart, new_name.c_str());
  }

  return result;
}

extern "C" {

void mpi_init_(int *ierror) {
  pmpi_init_(ierror);
  if (*ierror != MPI_SUCCESS)
    return;

  int world_rank = 0;
  int world_size = 0;
  int f_ierr = 0;
  MPI_Fint f_comm_world = MPI_Comm_c2f(MPI_COMM_WORLD);

  pmpi_comm_rank_(&f_comm_world, &world_rank, &f_ierr);
  pmpi_comm_size_(&f_comm_world, &world_size, &f_ierr);

  std::string world_name = get_fortran_communicator_name(f_comm_world);

  Profiler::instance().initialise(world_rank, world_size, world_name);
}

void mpi_init_thread_(int *required, int *provided, int *ierror) {
  pmpi_init_thread_(required, provided, ierror);
  if (*ierror != MPI_SUCCESS)
    return;

  int world_rank = 0;
  int world_size = 0;
  int f_ierr = 0;
  MPI_Fint f_comm_world = MPI_Comm_c2f(MPI_COMM_WORLD);

  pmpi_comm_rank_(&f_comm_world, &world_rank, &f_ierr);
  pmpi_comm_size_(&f_comm_world, &world_size, &f_ierr);

  std::string world_name = get_fortran_communicator_name(f_comm_world);

  Profiler::instance().initialise(world_rank, world_size, world_name);
}

void mpi_finalize_(int *ierror) {
  Profiler::instance().dump_profile_fortran();
  MPI_Fint f_comm_world = MPI_Comm_c2f(MPI_COMM_WORLD);
  int f_ierr;
  pmpi_barrier_(&f_comm_world, &f_ierr);
  pmpi_finalize_(ierror);
}

void mpi_comm_split_(MPI_Fint *comm, int *color, int *key, MPI_Fint *newcomm,
                     int *ierror) {
  pmpi_comm_split_(comm, color, key, newcomm, ierror);

  if (*ierror == MPI_SUCCESS) {
    std::string old_name = get_fortran_communicator_name(*comm);

    int new_size = 0;
    int f_ierr = 0;
    pmpi_comm_size_(newcomm, &new_size, &f_ierr);

    std::vector<int> world_ranks_array(new_size);
    int dynamic_rank = Profiler::instance().get_global_rank().value_or(0);
    int sendcount = 1;

    MPI_Fint f_int_type = MPI_Type_c2f(MPI_INT);

    pmpi_allgather_(&dynamic_rank, &sendcount, &f_int_type,
                    world_ranks_array.data(), &sendcount, &f_int_type, newcomm,
                    &f_ierr);

    std::string new_name =
        Profiler::instance().generate_and_track_communicator(
            old_name, "split", *color, world_ranks_array);

    int name_len = new_name.length();
    pmpi_comm_set_name_(newcomm, const_cast<char *>(new_name.c_str()), &f_ierr,
                        name_len);
  }
}

void mpi_cart_create_(MPI_Fint *comm_old, int *ndims, int *dims, int *periods,
                      int *reorder, MPI_Fint *comm_cart_f, int *ierror) {
  pmpi_cart_create_(comm_old, ndims, dims, periods, reorder, comm_cart_f,
                    ierror);
  if (*ierror != MPI_SUCCESS)
    return;

  std::string old_name = get_fortran_communicator_name(*comm_old);

  int new_size;
  MPI_Fint f_cart_comm = *comm_cart_f;
  int f_ierr = 0;
  pmpi_comm_size_(&f_cart_comm, &new_size, &f_ierr);
  if (f_ierr != MPI_SUCCESS) {
    *ierror = f_ierr;
    return;
  }

  std::vector<int> world_ranks_array(new_size);
  int dynamic_rank = Profiler::instance().get_global_rank().value_or(0);
  int sendcount = 1, recvcount = 1;
  MPI_Fint f_int_type = MPI_Type_c2f(MPI_INT);

  pmpi_allgather_(&dynamic_rank, &sendcount, &f_int_type,
                  world_ranks_array.data(), &recvcount, &f_int_type,
                  &f_cart_comm, &f_ierr);
  if (f_ierr != MPI_SUCCESS) {
    *ierror = f_ierr;
    return;
  }

  std::string new_name =
      Profiler::instance().generate_and_track_communicator(
          old_name, "cart", 0, world_ranks_array);

  pmpi_comm_set_name_(comm_cart_f, const_cast<char *>(new_name.c_str()), ierror,
                      new_name.length());
}

void mpi_comm_dup_(MPI_Fint *comm, MPI_Fint *newcomm_f, int *ierror) {
  pmpi_comm_dup_(comm, newcomm_f, ierror);
  if (*ierror != MPI_SUCCESS)
    return;

  std::string old_name = get_fortran_communicator_name(*comm);

  int new_size;
  int f_ierr = 0;
  pmpi_comm_size_(newcomm_f, &new_size, &f_ierr);
  if (f_ierr != MPI_SUCCESS) {
    *ierror = f_ierr;
    return;
  }

  std::vector<int> world_ranks_array(new_size);
  int dynamic_rank = Profiler::instance().get_global_rank().value_or(0);
  int sendcount = 1, recvcount = 1;
  MPI_Fint f_int_type = MPI_Type_c2f(MPI_INT);

  pmpi_allgather_(&dynamic_rank, &sendcount, &f_int_type,
                  world_ranks_array.data(), &recvcount, &f_int_type, newcomm_f,
                  &f_ierr);
  if (f_ierr != MPI_SUCCESS) {
    *ierror = f_ierr;
    return;
  }

  std::string new_name =
      Profiler::instance().generate_and_track_communicator(
          old_name, "dup", 0, world_ranks_array);

  pmpi_comm_set_name_(newcomm_f, const_cast<char *>(new_name.c_str()), ierror,
                      new_name.length());
}
}

extern "C" {

// point to point
void mpi_send_(void *buf, int *count, MPI_Fint *datatype, int *dest, int *tag,
               MPI_Fint *comm, int *ierror) {
  MPI_Comm c_comm = MPI_Comm_f2c(*comm);
  Profiler::instance().record_p2p_send(*dest,
                                            get_communicator_name(c_comm));
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
  Profiler::instance().record_p2p_send(*dest,
                                            get_communicator_name(c_comm));
  pmpi_sendrecv_(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf,
                 recvcount, recvtype, source, recvtag, comm, status_f, ierror);
}

void mpi_isend_(void *buf, int *count, MPI_Fint *datatype, int *dest, int *tag,
                MPI_Fint *comm, MPI_Fint *request_f, int *ierror) {
  MPI_Comm c_comm = MPI_Comm_f2c(*comm);
  Profiler::instance().record_p2p_send(*dest,
                                            get_communicator_name(c_comm));
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

// collective
void mpi_bcast_(void *buffer, int *count, MPI_Fint *datatype, int *root,
                MPI_Fint *comm, int *ierror) {
  MPI_Comm c_comm = MPI_Comm_f2c(*comm);
  Profiler::instance().record_collective_broadcast(
      get_communicator_name(c_comm));
  pmpi_bcast_(buffer, count, datatype, root, comm, ierror);
}

void mpi_reduce_(const void *sendbuf, void *recvbuf, int *count,
                 MPI_Fint *datatype, MPI_Fint *op, int *root, MPI_Fint *comm,
                 int *ierror) {
  MPI_Comm c_comm = MPI_Comm_f2c(*comm);
  // TODO: Investigate how I should record a reduction
  Profiler::instance().record_collective_broadcast(
      get_communicator_name(c_comm));
  pmpi_reduce_(sendbuf, recvbuf, count, datatype, op, root, comm, ierror);
}

void mpi_allreduce_(const void *sendbuf, void *recvbuf, int *count,
                    MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm,
                    int *ierror) {
  MPI_Comm c_comm = MPI_Comm_f2c(*comm);
  Profiler::instance().record_collective_broadcast(
      get_communicator_name(c_comm));
  pmpi_allreduce_(sendbuf, recvbuf, count, datatype, op, comm, ierror);
}

void mpi_scatter_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype,
                  void *recvbuf, int *recvcount, MPI_Fint *recvtype, int *root,
                  MPI_Fint *comm, int *ierror) {
  MPI_Comm c_comm = MPI_Comm_f2c(*comm);
  Profiler::instance().record_collective_broadcast(
      get_communicator_name(c_comm));
  pmpi_scatter_(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                root, comm, ierror);
}

void mpi_gather_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype,
                 void *recvbuf, int *recvcount, MPI_Fint *recvtype, int *root,
                 MPI_Fint *comm, int *ierror) {
  MPI_Comm c_comm = MPI_Comm_f2c(*comm);
  Profiler::instance().record_collective_broadcast(
      get_communicator_name(c_comm));
  pmpi_gather_(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
               comm, ierror);
}

void mpi_allgather_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype,
                    void *recvbuf, int *recvcount, MPI_Fint *recvtype,
                    MPI_Fint *comm, int *ierror) {
  MPI_Comm c_comm = MPI_Comm_f2c(*comm);
  Profiler::instance().record_collective_broadcast(
      get_communicator_name(c_comm));
  pmpi_allgather_(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                  comm, ierror);
}

void mpi_alltoall_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype,
                   void *recvbuf, int *recvcount, MPI_Fint *recvtype,
                   MPI_Fint *comm, int *ierror) {
  MPI_Comm c_comm = MPI_Comm_f2c(*comm);
  Profiler::instance().record_collective_broadcast(
      get_communicator_name(c_comm));
  pmpi_alltoall_(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                 comm, ierror);
}

void mpi_gatherv_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype,
                  void *recvbuf, int *recvcounts, int *displs,
                  MPI_Fint *recvtype, int *root, MPI_Fint *comm, int *ierror) {
  MPI_Comm c_comm = MPI_Comm_f2c(*comm);
  Profiler::instance().record_collective_broadcast(
      get_communicator_name(c_comm));
  pmpi_gatherv_(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs,
                recvtype, root, comm, ierror);
}

void mpi_scatterv_(const void *sendbuf, int *sendcounts, int *displs,
                   MPI_Fint *sendtype, void *recvbuf, int *recvcount,
                   MPI_Fint *recvtype, int *root, MPI_Fint *comm, int *ierror) {
  MPI_Comm c_comm = MPI_Comm_f2c(*comm);
  Profiler::instance().record_collective_broadcast(
      get_communicator_name(c_comm));
  pmpi_scatterv_(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount,
                 recvtype, root, comm, ierror);
}

void mpi_allgatherv_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype,
                     void *recvbuf, int *recvcounts, int *displs,
                     MPI_Fint *recvtype, MPI_Fint *comm, int *ierror) {
  MPI_Comm c_comm = MPI_Comm_f2c(*comm);
  Profiler::instance().record_collective_broadcast(
      get_communicator_name(c_comm));
  pmpi_allgatherv_(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs,
                   recvtype, comm, ierror);
}

void mpi_alltoallv_(const void *sendbuf, int *sendcounts, int *sdispls,
                    MPI_Fint *sendtype, void *recvbuf, int *recvcounts,
                    int *rdispls, MPI_Fint *recvtype, MPI_Fint *comm,
                    int *ierror) {
  MPI_Comm c_comm = MPI_Comm_f2c(*comm);
  Profiler::instance().record_collective_broadcast(
      get_communicator_name(c_comm));
  pmpi_alltoallv_(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts,
                  rdispls, recvtype, comm, ierror);
}

void mpi_barrier_(MPI_Fint *comm, int *ierror) {
  MPI_Comm c_comm = MPI_Comm_f2c(*comm);
  Profiler::instance().record_collective_broadcast(
      get_communicator_name(c_comm));
  pmpi_barrier_(comm, ierror);
}
}
