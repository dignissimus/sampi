#include "rank_reorder.h"
#include "fortran_headers.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <mpi.h>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

// TODO: Read error value
int MPI_Init(int *argc, char ***argv) {
  int return_value = PMPI_Init(argc, argv);
  if (return_value != MPI_SUCCESS)
    return return_value;

  int world_rank, world_size;
  PMPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &world_size);

  MPI_Comm new_comm;
  int err = PMPI_Comm_split(MPI_COMM_WORLD, 0, world_rank, &new_comm);
  if (err == MPI_SUCCESS) {
    RankReorder::instance().set_reorder_comm_world(new_comm);
  }
  return return_value;
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided) {
  int return_value = PMPI_Init_thread(argc, argv, required, provided);
  if (return_value != MPI_SUCCESS)
    return return_value;

  int world_rank, world_size;
  PMPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &world_size);

  MPI_Comm new_comm;
  int err = PMPI_Comm_split(MPI_COMM_WORLD, 0, world_rank, &new_comm);

  if (err == MPI_SUCCESS) {
    RankReorder::instance().set_reorder_comm_world(new_comm);
  }

  return return_value;
}

int MPI_Finalize() {
  MPI_Comm current_reorder = RankReorder::instance().get_reorder_comm_world();

  if (current_reorder != MPI_COMM_NULL) {
    PMPI_Comm_free(&current_reorder);
    RankReorder::instance().set_reorder_comm_world(MPI_COMM_NULL);
  }

  return PMPI_Finalize();
}

int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm) {
  return PMPI_Comm_dup(RankReorder::instance().reorder_communicator(comm),
                       newcomm);
}

int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm) {
  return PMPI_Comm_split(RankReorder::instance().reorder_communicator(comm),
                         color, key, newcomm);
}

int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
             int tag, MPI_Comm comm) {
  return PMPI_Send(buf, count, datatype, dest, tag,
                   RankReorder::instance().reorder_communicator(comm));
}

int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
             MPI_Comm comm, MPI_Status *status) {
  return PMPI_Recv(buf, count, datatype, source, tag,
                   RankReorder::instance().reorder_communicator(comm), status);
}

int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 int dest, int sendtag, void *recvbuf, int recvcount,
                 MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm,
                 MPI_Status *status) {

  return PMPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf,
                       recvcount, recvtype, source, recvtag,
                       RankReorder::instance().reorder_communicator(comm),
                       status);
}

int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm, MPI_Request *request) {
  return PMPI_Isend(buf, count, datatype, dest, tag,
                    RankReorder::instance().reorder_communicator(comm),
                    request);
}

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
              MPI_Comm comm, MPI_Request *request) {
  return PMPI_Irecv(buf, count, datatype, source, tag,
                    RankReorder::instance().reorder_communicator(comm),
                    request);
}

int MPI_Wait(MPI_Request *request, MPI_Status *status) {
  return PMPI_Wait(request, status);
}

int MPI_Waitall(int count, MPI_Request array_of_requests[],
                MPI_Status array_of_statuses[]) {
  return PMPI_Waitall(count, array_of_requests, array_of_statuses);
}

int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root,
              MPI_Comm comm) {
  return PMPI_Bcast(buffer, count, datatype, root,
                    RankReorder::instance().reorder_communicator(comm));
}

int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
               MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
  return PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root,
                     RankReorder::instance().reorder_communicator(comm));
}

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op,
                        RankReorder::instance().reorder_communicator(comm));
}

int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                MPI_Comm comm) {
  return PMPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                      recvtype, root,
                      RankReorder::instance().reorder_communicator(comm));
}

int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
               MPI_Comm comm) {
  return PMPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                     root, RankReorder::instance().reorder_communicator(comm));
}

int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  MPI_Comm comm) {
  return PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                        recvtype,
                        RankReorder::instance().reorder_communicator(comm));
}

int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 MPI_Comm comm) {
  return PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                       recvtype,
                       RankReorder::instance().reorder_communicator(comm));
}

int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, const int recvcounts[], const int displs[],
                MPI_Datatype recvtype, int root, MPI_Comm comm) {
  return PMPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs,
                      recvtype, root,
                      RankReorder::instance().reorder_communicator(comm));
}

int MPI_Scatterv(const void *sendbuf, const int sendcounts[],
                 const int displs[], MPI_Datatype sendtype, void *recvbuf,
                 int recvcount, MPI_Datatype recvtype, int root,
                 MPI_Comm comm) {
  return PMPI_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf,
                       recvcount, recvtype, root,
                       RankReorder::instance().reorder_communicator(comm));
}

int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, const int recvcounts[], const int displs[],
                   MPI_Datatype recvtype, MPI_Comm comm) {
  return PMPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts,
                         displs, recvtype,
                         RankReorder::instance().reorder_communicator(comm));
}

int MPI_Alltoallv(const void *sendbuf, const int sendcounts[],
                  const int sdispls[], MPI_Datatype sendtype, void *recvbuf,
                  const int recvcounts[], const int rdispls[],
                  MPI_Datatype recvtype, MPI_Comm comm) {
  return PMPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf,
                        recvcounts, rdispls, recvtype,
                        RankReorder::instance().reorder_communicator(comm));
}

int MPI_Barrier(MPI_Comm comm) {
  return PMPI_Barrier(RankReorder::instance().reorder_communicator(comm));
}

int MPI_Comm_rank(MPI_Comm comm, int *rank) {
  int ret =
      PMPI_Comm_rank(RankReorder::instance().reorder_communicator(comm), rank);
  return ret;
}

int MPI_Comm_size(MPI_Comm comm, int *size) {
  int ret =
      PMPI_Comm_size(RankReorder::instance().reorder_communicator(comm), size);
  return ret;
}

int MPI_Cart_create(MPI_Comm comm_old, int ndims, const int dims[],
                    const int periods[], int reorder, MPI_Comm *comm_cart) {

  MPI_Comm resolved_comm =
      RankReorder::instance().reorder_communicator(comm_old);

  return PMPI_Cart_create(resolved_comm, ndims, dims, periods, reorder,
                          comm_cart);
}

// todo: deduplicate initialisation logic
extern "C" {

void mpi_init_(int *ierror) {
  pmpi_init_(ierror);
  if (*ierror != MPI_SUCCESS)
    return;

  int world_rank, world_size;
  MPI_Fint f_comm_world = MPI_Comm_c2f(MPI_COMM_WORLD);

  pmpi_comm_rank_(&f_comm_world, &world_rank, ierror);
  if (*ierror != MPI_SUCCESS)
    return;

  pmpi_comm_size_(&f_comm_world, &world_size, ierror);
  if (*ierror != MPI_SUCCESS)
    return;

  MPI_Fint f_new_comm;
  int colour = 0;

  pmpi_comm_split_(&f_comm_world, &colour, &world_rank, &f_new_comm, ierror);

  if (*ierror == MPI_SUCCESS) {
    RankReorder::instance().set_reorder_comm_world(MPI_Comm_f2c(f_new_comm));
  }
}

void mpi_init_thread_(int *required, int *provided, int *ierror) {
  pmpi_init_thread_(required, provided, ierror);
  if (*ierror != MPI_SUCCESS)
    return;

  int world_rank, world_size;
  MPI_Fint f_comm_world = MPI_Comm_c2f(MPI_COMM_WORLD);

  pmpi_comm_rank_(&f_comm_world, &world_rank, ierror);
  if (*ierror != MPI_SUCCESS)
    return;

  pmpi_comm_size_(&f_comm_world, &world_size, ierror);
  if (*ierror != MPI_SUCCESS)
    return;

  MPI_Fint f_new_comm;
  int colour = 0;

  pmpi_comm_split_(&f_comm_world, &colour, &world_rank, &f_new_comm, ierror);

  if (*ierror == MPI_SUCCESS) {
    RankReorder::instance().set_reorder_comm_world(MPI_Comm_f2c(f_new_comm));
  }
}

void mpi_finalize_(int *ierror) {
  MPI_Comm current_reorder = RankReorder::instance().get_reorder_comm_world();

  if (current_reorder != MPI_COMM_NULL) {
    MPI_Fint f_reorder_comm = MPI_Comm_c2f(current_reorder);
    int _ierr = 0;
    pmpi_comm_free_(&f_reorder_comm, &_ierr);
    RankReorder::instance().set_reorder_comm_world(MPI_COMM_NULL);
  }

  pmpi_finalize_(ierror);
}

void mpi_send_(void *buf, int *count, MPI_Fint *datatype, int *dest, int *tag,
               MPI_Fint *comm, int *ierror) {
  MPI_Fint reordered_comm =
      RankReorder::instance().reorder_communicator_fortran(*comm);
  pmpi_send_(buf, count, datatype, dest, tag, &reordered_comm, ierror);
}

void mpi_recv_(void *buf, int *count, MPI_Fint *datatype, int *source, int *tag,
               MPI_Fint *comm, MPI_Fint *status_f, int *ierror) {
  MPI_Fint reordered_comm =
      RankReorder::instance().reorder_communicator_fortran(*comm);
  pmpi_recv_(buf, count, datatype, source, tag, &reordered_comm, status_f,
             ierror);
}

void mpi_sendrecv_(void *sendbuf, int *sendcount, MPI_Fint *sendtype, int *dest,
                   int *sendtag, void *recvbuf, int *recvcount,
                   MPI_Fint *recvtype, int *source, int *recvtag,
                   MPI_Fint *comm, MPI_Fint *status_f, int *ierror) {
  MPI_Fint reordered_comm =
      RankReorder::instance().reorder_communicator_fortran(*comm);
  pmpi_sendrecv_(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf,
                 recvcount, recvtype, source, recvtag, &reordered_comm,
                 status_f, ierror);
}

void mpi_isend_(void *buf, int *count, MPI_Fint *datatype, int *dest, int *tag,
                MPI_Fint *comm, MPI_Fint *request_f, int *ierror) {
  MPI_Fint reordered_comm =
      RankReorder::instance().reorder_communicator_fortran(*comm);
  pmpi_isend_(buf, count, datatype, dest, tag, &reordered_comm, request_f,
              ierror);
}

void mpi_irecv_(void *buf, int *count, MPI_Fint *datatype, int *source,
                int *tag, MPI_Fint *comm, MPI_Fint *request_f, int *ierror) {
  MPI_Fint reordered_comm =
      RankReorder::instance().reorder_communicator_fortran(*comm);
  pmpi_irecv_(buf, count, datatype, source, tag, &reordered_comm, request_f,
              ierror);
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
  MPI_Fint reordered_comm =
      RankReorder::instance().reorder_communicator_fortran(*comm);
  pmpi_bcast_(buffer, count, datatype, root, &reordered_comm, ierror);
}

void mpi_reduce_(const void *sendbuf, void *recvbuf, int *count,
                 MPI_Fint *datatype, MPI_Fint *op, int *root, MPI_Fint *comm,
                 int *ierror) {
  MPI_Fint reordered_comm =
      RankReorder::instance().reorder_communicator_fortran(*comm);
  pmpi_reduce_(sendbuf, recvbuf, count, datatype, op, root, &reordered_comm,
               ierror);
}

void mpi_allreduce_(const void *sendbuf, void *recvbuf, int *count,
                    MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm,
                    int *ierror) {
  MPI_Fint reordered_comm =
      RankReorder::instance().reorder_communicator_fortran(*comm);
  pmpi_allreduce_(sendbuf, recvbuf, count, datatype, op, &reordered_comm,
                  ierror);
}

void mpi_scatter_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype,
                  void *recvbuf, int *recvcount, MPI_Fint *recvtype, int *root,
                  MPI_Fint *comm, int *ierror) {
  MPI_Fint reordered_comm =
      RankReorder::instance().reorder_communicator_fortran(*comm);
  pmpi_scatter_(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                root, &reordered_comm, ierror);
}

void mpi_gather_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype,
                 void *recvbuf, int *recvcount, MPI_Fint *recvtype, int *root,
                 MPI_Fint *comm, int *ierror) {
  MPI_Fint reordered_comm =
      RankReorder::instance().reorder_communicator_fortran(*comm);
  pmpi_gather_(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
               &reordered_comm, ierror);
}

void mpi_allgather_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype,
                    void *recvbuf, int *recvcount, MPI_Fint *recvtype,
                    MPI_Fint *comm, int *ierror) {
  MPI_Fint reordered_comm =
      RankReorder::instance().reorder_communicator_fortran(*comm);
  pmpi_allgather_(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                  &reordered_comm, ierror);
}

void mpi_alltoall_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype,
                   void *recvbuf, int *recvcount, MPI_Fint *recvtype,
                   MPI_Fint *comm, int *ierror) {
  MPI_Fint reordered_comm =
      RankReorder::instance().reorder_communicator_fortran(*comm);
  pmpi_alltoall_(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                 &reordered_comm, ierror);
}

void mpi_gatherv_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype,
                  void *recvbuf, int *recvcounts, int *displs,
                  MPI_Fint *recvtype, int *root, MPI_Fint *comm, int *ierror) {
  MPI_Fint reordered_comm =
      RankReorder::instance().reorder_communicator_fortran(*comm);
  pmpi_gatherv_(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs,
                recvtype, root, &reordered_comm, ierror);
}

void mpi_scatterv_(const void *sendbuf, int *sendcounts, int *displs,
                   MPI_Fint *sendtype, void *recvbuf, int *recvcount,
                   MPI_Fint *recvtype, int *root, MPI_Fint *comm, int *ierror) {
  MPI_Fint reordered_comm =
      RankReorder::instance().reorder_communicator_fortran(*comm);
  pmpi_scatterv_(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount,
                 recvtype, root, &reordered_comm, ierror);
}

void mpi_allgatherv_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype,
                     void *recvbuf, int *recvcounts, int *displs,
                     MPI_Fint *recvtype, MPI_Fint *comm, int *ierror) {
  MPI_Fint reordered_comm =
      RankReorder::instance().reorder_communicator_fortran(*comm);
  pmpi_allgatherv_(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs,
                   recvtype, &reordered_comm, ierror);
}

void mpi_alltoallv_(const void *sendbuf, int *sendcounts, int *sdispls,
                    MPI_Fint *sendtype, void *recvbuf, int *recvcounts,
                    int *rdispls, MPI_Fint *recvtype, MPI_Fint *comm,
                    int *ierror) {
  MPI_Fint reordered_comm =
      RankReorder::instance().reorder_communicator_fortran(*comm);
  pmpi_alltoallv_(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts,
                  rdispls, recvtype, &reordered_comm, ierror);
}

void mpi_barrier_(MPI_Fint *comm, int *ierror) {
  MPI_Fint reordered_comm =
      RankReorder::instance().reorder_communicator_fortran(*comm);
  pmpi_barrier_(&reordered_comm, ierror);
}

void mpi_comm_rank_(MPI_Fint *comm, int *rank, int *ierror) {
  MPI_Fint reordered_comm =
      RankReorder::instance().reorder_communicator_fortran(*comm);
  pmpi_comm_rank_(&reordered_comm, rank, ierror);
}

void mpi_comm_size_(MPI_Fint *comm, int *size, int *ierror) {
  MPI_Fint reordered_comm =
      RankReorder::instance().reorder_communicator_fortran(*comm);
  pmpi_comm_size_(&reordered_comm, size, ierror);
}

void mpi_cart_create_(MPI_Fint *comm_old, int *ndims, int *dims, int *periods,
                      int *reorder_f, MPI_Fint *comm_cart_f, int *ierror) {
  MPI_Fint reordered_comm =
      RankReorder::instance().reorder_communicator_fortran(*comm_old);
  pmpi_cart_create_(&reordered_comm, ndims, dims, periods, reorder_f,
                    comm_cart_f, ierror);
}

void mpi_comm_dup_(MPI_Fint *comm, MPI_Fint *newcomm_f, int *ierror) {
  MPI_Fint reordered_comm_old =
      RankReorder::instance().reorder_communicator_fortran(*comm);
  pmpi_comm_dup_(&reordered_comm_old, newcomm_f, ierror);
}

void mpi_comm_split_(MPI_Fint *comm, int *color, int *key, MPI_Fint *newcomm_f,
                     int *ierror) {
  MPI_Fint reordered_comm_old =
      RankReorder::instance().reorder_communicator_fortran(*comm);
  pmpi_comm_split_(&reordered_comm_old, color, key, newcomm_f, ierror);
}
}
