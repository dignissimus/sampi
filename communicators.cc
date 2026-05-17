#include "communicators.h"
#include <string>
#include <iostream>

std::string get_communicator_name(const MPI_Comm &comm) {
  char comm_name[MPI_MAX_OBJECT_NAME];
  int comm_name_length;
  PMPI_Comm_get_name(comm, comm_name, &comm_name_length);
  return std::string(comm_name, comm_name_length);
}

std::string get_fortran_communicator_name(MPI_Fint f_comm) {
  char comm_name[MPI_MAX_OBJECT_NAME];
  int comm_name_length = 0;
  int ierr = 0;
  pmpi_comm_get_name_(&f_comm, comm_name, &comm_name_length, &ierr,
                      MPI_MAX_OBJECT_NAME);

  if (ierr != MPI_SUCCESS || comm_name_length == 0) {
    std::cerr << "Warning: Unable to get name for Fortran communicator with ID "
              << f_comm << std::endl;
    return std::string("UNKNOWN_FORTRAN_COMMUNICATOR");
  }
  return std::string(comm_name, comm_name_length);
}
