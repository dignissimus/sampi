#include <mpi.h>

extern "C" {
  void pmpi_init_(int *ierror);
  void pmpi_init_thread_(int *required, int *provided, int *ierror);
  void pmpi_finalize_(int *ierror);
  void pmpi_send_(void *buf, int *count, MPI_Fint *datatype, int *dest, int *tag, MPI_Fint *comm, int *ierror);
  void pmpi_recv_(void *buf, int *count, MPI_Fint *datatype, int *source, int *tag, MPI_Fint *comm, MPI_Fint *status_f, int *ierror);
  void pmpi_sendrecv_(
    void *sendbuf, int *sendcount, MPI_Fint *sendtype, int *dest, int *sendtag,
    void *recvbuf, int *recvcount, MPI_Fint *recvtype, int *source, int *recvtag,
    MPI_Fint *comm, MPI_Fint *status_f, int *ierror
  );
  void pmpi_isend_(void *buf, int *count, MPI_Fint *datatype, int *dest, int *tag, MPI_Fint *comm, MPI_Fint *request_f, int *ierror);
  void pmpi_irecv_(void *buf, int *count, MPI_Fint *datatype, int *source, int *tag, MPI_Fint *comm, MPI_Fint *request_f, int *ierror);
  void pmpi_wait_(MPI_Fint *request_f, MPI_Fint *status_f, int *ierror);
  void pmpi_waitall_(int *count, MPI_Fint *requests_f, MPI_Fint *statuses_f, int *ierror);
  void pmpi_bcast_(void *buffer, int *count, MPI_Fint *datatype, int *root, MPI_Fint *comm, int *ierror);
  void pmpi_reduce_(const void *sendbuf, void *recvbuf, int *count, MPI_Fint *datatype, MPI_Fint *op, int *root, MPI_Fint *comm, int *ierror);
  void pmpi_allreduce_(const void *sendbuf, void *recvbuf, int *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, int *ierror);
  void pmpi_scatter_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype, void *recvbuf, int *recvcount, MPI_Fint *recvtype, int *root, MPI_Fint *comm, int *ierror);
  void pmpi_gather_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype, void *recvbuf, int *recvcount, MPI_Fint *recvtype, int *root, MPI_Fint *comm, int *ierror);
  void pmpi_allgather_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype, void *recvbuf, int *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, int *ierror);
  void pmpi_alltoall_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype, void *recvbuf, int *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, int *ierror);
  void pmpi_comm_split_(MPI_Fint *comm, int *color, int *key, MPI_Fint *newcomm_f, int *ierror);
  void pmpi_cart_create_(MPI_Fint *comm_old, int *ndims, int *dims, int *periods, int *reorder, MPI_Fint *comm_cart_f, int *ierror);

  void pmpi_comm_rank_(MPI_Fint *comm, int *rank, int *ierror);
  void pmpi_comm_size_(MPI_Fint *comm, int *size, int *ierror);
  void pmpi_barrier_(MPI_Fint *comm, int *ierror);
  void pmpi_comm_get_name_(MPI_Fint *comm, char *name, int *len, int *ierror, int name_len);
  void pmpi_comm_set_name_(MPI_Fint *comm, char *name, int *ierror, int name_len);
  void pmpi_comm_dup_(MPI_Fint *comm, MPI_Fint *newcomm_f, int *ierror);
  void pmpi_comm_free_(MPI_Fint *comm, int *ierror);
  void pmpi_gatherv_(
      const void *sendbuf, int *sendcount, MPI_Fint *sendtype,
      void *recvbuf, int *recvcounts, int *displs, MPI_Fint *recvtype,
      int *root, MPI_Fint *comm, int *ierror
  );
  void pmpi_scatterv_(
      const void *sendbuf, int *sendcounts, int *displs, MPI_Fint *sendtype,
      void *recvbuf, int *recvcount, MPI_Fint *recvtype,
      int *root, MPI_Fint *comm, int *ierror
  );
  void pmpi_allgatherv_(
      const void *sendbuf, int *sendcount, MPI_Fint *sendtype,
      void *recvbuf, int *recvcounts, int *displs, MPI_Fint *recvtype,
      MPI_Fint *comm, int *ierror
  );
  void pmpi_alltoallv_(
      const void *sendbuf, int *sendcounts, int *sdispls, MPI_Fint *sendtype,
      void *recvbuf, int *recvcounts, int *rdispls, MPI_Fint *recvtype,
      MPI_Fint *comm, int *ierror
  );
}
