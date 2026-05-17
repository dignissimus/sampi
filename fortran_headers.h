#include <mpi.h>

#define DECLARE_PMPI_VARIANTS(lower, upper, args_definition) \
    extern "C" void pmpi_##lower##_ args_definition; \
    extern "C" void pmpi_##lower##__ args_definition; \
    extern "C" void pmpi_##lower args_definition; \
    extern "C" void PMPI_##upper args_definition;

// TODO: Can I automatically generate these from mpi.h?
extern "C" {
    DECLARE_PMPI_VARIANTS(init, INIT, (int *ierror))
    DECLARE_PMPI_VARIANTS(init_thread, INIT_THREAD, (int *required, int *provided, int *ierror))
    DECLARE_PMPI_VARIANTS(finalize, FINALIZE, (int *ierror))
    DECLARE_PMPI_VARIANTS(abort, ABORT, (MPI_Fint *comm, int *errorcode, int *ierror))
    DECLARE_PMPI_VARIANTS(initialized, INITIALIZED, (int *flag, int *ierror))
    DECLARE_PMPI_VARIANTS(finalized, FINALIZED, (int *flag, int *ierror))

    DECLARE_PMPI_VARIANTS(comm_rank, COMM_RANK, (MPI_Fint *comm, int *rank, int *ierror))
    DECLARE_PMPI_VARIANTS(comm_size, COMM_SIZE, (MPI_Fint *comm, int *size, int *ierror))
    DECLARE_PMPI_VARIANTS(comm_dup, COMM_DUP, (MPI_Fint *comm, MPI_Fint *newcomm_f, int *ierror))
    DECLARE_PMPI_VARIANTS(comm_free, COMM_FREE, (MPI_Fint *comm, int *ierror))
    DECLARE_PMPI_VARIANTS(comm_split, COMM_SPLIT, (MPI_Fint *comm, int *color, int *key, MPI_Fint *newcomm_f, int *ierror))
    DECLARE_PMPI_VARIANTS(comm_split_type, COMM_SPLIT_TYPE, (MPI_Fint *comm, int *split_type, int *key, MPI_Fint *info, MPI_Fint *newcomm_f, int *ierror))
    DECLARE_PMPI_VARIANTS(comm_create, COMM_CREATE, (MPI_Fint *comm, MPI_Fint *group, MPI_Fint *newcomm_f, int *ierror))
    DECLARE_PMPI_VARIANTS(comm_create_group, COMM_CREATE_GROUP, (MPI_Fint *comm, MPI_Fint *group, int *tag, MPI_Fint *newcomm_f, int *ierror))
    DECLARE_PMPI_VARIANTS(cart_create, CART_CREATE, (MPI_Fint *comm_old, int *ndims, int *dims, int *periods, int *reorder, MPI_Fint *comm_cart_f, int *ierror))
    DECLARE_PMPI_VARIANTS(comm_get_name, COMM_GET_NAME, (MPI_Fint *comm, char *name, int *len, int *ierror, int name_len))
    DECLARE_PMPI_VARIANTS(comm_set_name, COMM_SET_NAME, (MPI_Fint *comm, char *name, int *ierror, int name_len))

    DECLARE_PMPI_VARIANTS(send, SEND, (const void *buf, int *count, MPI_Fint *datatype, int *dest, int *tag, MPI_Fint *comm, int *ierror))
    DECLARE_PMPI_VARIANTS(recv, RECV, (void *buf, int *count, MPI_Fint *datatype, int *source, int *tag, MPI_Fint *comm, MPI_Fint *status_f, int *ierror))
    DECLARE_PMPI_VARIANTS(sendrecv, SENDRECV, (const void *sendbuf, int *sendcount, MPI_Fint *sendtype, int *dest, int *sendtag, void *recvbuf, int *recvcount, MPI_Fint *recvtype, int *source, int *recvtag, MPI_Fint *comm, MPI_Fint *status_f, int *ierror))
    DECLARE_PMPI_VARIANTS(isend, ISEND, (const void *buf, int *count, MPI_Fint *datatype, int *dest, int *tag, MPI_Fint *comm, MPI_Fint *request_f, int *ierror))
    DECLARE_PMPI_VARIANTS(irecv, IRECV, (void *buf, int *count, MPI_Fint *datatype, int *source, int *tag, MPI_Fint *comm, MPI_Fint *request_f, int *ierror))
    DECLARE_PMPI_VARIANTS(wait, WAIT, (MPI_Fint *request_f, MPI_Fint *status_f, int *ierror))
    DECLARE_PMPI_VARIANTS(waitall, WAITALL, (int *count, MPI_Fint *requests_f, MPI_Fint *statuses_f, int *ierror))
    DECLARE_PMPI_VARIANTS(probe, PROBE, (int *source, int *tag, MPI_Fint *comm, MPI_Fint *status_f, int *ierror))
    DECLARE_PMPI_VARIANTS(iprobe, IPROBE, (int *source, int *tag, MPI_Fint *comm, int *flag, MPI_Fint *status_f, int *ierror))
    DECLARE_PMPI_VARIANTS(test, TEST, (MPI_Fint *request_f, int *flag, MPI_Fint *status_f, int *ierror))

    DECLARE_PMPI_VARIANTS(barrier, BARRIER, (MPI_Fint *comm, int *ierror))
    DECLARE_PMPI_VARIANTS(bcast, BCAST, (void *buffer, int *count, MPI_Fint *datatype, int *root, MPI_Fint *comm, int *ierror))
    DECLARE_PMPI_VARIANTS(reduce, REDUCE, (const void *sendbuf, void *recvbuf, int *count, MPI_Fint *datatype, MPI_Fint *op, int *root, MPI_Fint *comm, int *ierror))
    DECLARE_PMPI_VARIANTS(allreduce, ALLREDUCE, (const void *sendbuf, void *recvbuf, int *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, int *ierror))
    DECLARE_PMPI_VARIANTS(scatter, SCATTER, (const void *sendbuf, int *sendcount, MPI_Fint *sendtype, void *recvbuf, int *recvcount, MPI_Fint *recvtype, int *root, MPI_Fint *comm, int *ierror))
    DECLARE_PMPI_VARIANTS(gather, GATHER, (const void *sendbuf, int *sendcount, MPI_Fint *sendtype, void *recvbuf, int *recvcount, MPI_Fint *recvtype, int *root, MPI_Fint *comm, int *ierror))
    DECLARE_PMPI_VARIANTS(allgather, ALLGATHER, (const void *sendbuf, int *sendcount, MPI_Fint *sendtype, void *recvbuf, int *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, int *ierror))
    DECLARE_PMPI_VARIANTS(alltoall, ALLTOALL, (const void *sendbuf, int *sendcount, MPI_Fint *sendtype, void *recvbuf, int *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, int *ierror))
    DECLARE_PMPI_VARIANTS(reduce_scatter, REDUCE_SCATTER, (const void *sendbuf, void *recvbuf, int *recvcounts, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, int *ierror))
    DECLARE_PMPI_VARIANTS(scan, SCAN, (const void *sendbuf, void *recvbuf, int *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, int *ierror))

    DECLARE_PMPI_VARIANTS(gatherv, GATHERV, (const void *sendbuf, int *sendcount, MPI_Fint *sendtype, void *recvbuf, int *recvcounts, int *displs, MPI_Fint *recvtype, int *root, MPI_Fint *comm, int *ierror))
    DECLARE_PMPI_VARIANTS(scatterv, SCATTERV, (const void *sendbuf, int *sendcounts, int *displs, MPI_Fint *sendtype, void *recvbuf, int *recvcount, MPI_Fint *recvtype, int *root, MPI_Fint *comm, int *ierror))
    DECLARE_PMPI_VARIANTS(allgatherv, ALLGATHERV, (const void *sendbuf, int *sendcount, MPI_Fint *sendtype, void *recvbuf, int *recvcounts, int *displs, MPI_Fint *recvtype, MPI_Fint *comm, int *ierror))
    DECLARE_PMPI_VARIANTS(alltoallv, ALLTOALLV, (const void *sendbuf, int *sendcounts, int *sdispls, MPI_Fint *sendtype, void *recvbuf, int *recvcounts, int *rdispls, MPI_Fint *recvtype, MPI_Fint *comm, int *ierror))
}
