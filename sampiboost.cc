#include "fortran_headers.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <mpi.h>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

static MPI_Comm reorder_comm_world = MPI_COMM_NULL;
static std::vector<int> rank_map;
struct PairHash {
  template <class T1, class T2>
  size_t operator()(const std::pair<T1, T2> &p) const {
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

using LatencyMapType =
    std::unordered_map<std::pair<int, int>, double, PairHash>;
using RankCommMapType = std::unordered_map<std::pair<int, int>, int, PairHash>;
// todo: shouldn't be global
static LatencyMapType latency_map(0, PairHash{});
static int global_rank;
#define reorder(comm) (comm == MPI_COMM_WORLD ? reorder_comm_world : comm)

// rename mapping, not clear that it's physical rank -> profiling rank
static double compute_cost(const std::vector<int> &mapping,
                           const LatencyMapType &latency_map,
                           const RankCommMapType &rank_comm, int size) {
  double cost = 0.0;
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      if (i == j)
        continue;
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

// todo: Probably have cost as a struct that has an update method
// todo: variable names
static double update_cost(const std::vector<int> &mapping,
                          const LatencyMapType &latency_map,
                          const RankCommMapType &rank_comm, int size, int si,
                          int sj, int original_cost) {
  double cost_difference = 0.0;

  int pi = mapping[si];
  int pj = mapping[sj];

  for (int x = 0; x < size; ++x) {
    auto px = mapping[x];

    double communication_cost_forward = rank_comm.at({pi, px});
    double communication_cost_backward = rank_comm.at({px, pi});
    cost_difference -= communication_cost_forward * latency_map.at({si, x});
    if(x != si) {
      cost_difference -= communication_cost_backward * latency_map.at({x, si});
    }

    communication_cost_forward = rank_comm.at({pj, px});
    communication_cost_backward = rank_comm.at({px, pj});
    cost_difference -= communication_cost_forward * latency_map.at({sj, x});
    if(x != sj) {
      cost_difference -= communication_cost_backward * latency_map.at({x, sj});
    }

  }

  for (int x = 0; x < size; ++x) {
    auto px = mapping[x];
    pi = mapping[sj];
    pj = mapping[si];

    double communication_cost_forward = rank_comm.at({pi, px});
    double communication_cost_backward = rank_comm.at({px, pi});
    cost_difference -= communication_cost_forward * latency_map.at({si, x});
    if(x != si) {
      cost_difference -= communication_cost_backward * latency_map.at({x, si});
    }

    communication_cost_forward = rank_comm.at({pj, px});
    communication_cost_backward = rank_comm.at({px, pj});
    cost_difference -= communication_cost_forward * latency_map.at({sj, x});
    if(x != sj) {
      cost_difference -= communication_cost_backward * latency_map.at({x, sj});
    }

  }
  return original_cost + cost_difference;
}

std::vector<int> compute_rank_map(int size, const LatencyMapType &latency_map,
                                  const RankCommMapType &rank_comm) {
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

    if (!improved)
      break;
  }

  return mapping;
}

std::vector<int> compute_rank_map_tree(int size,
                                       const LatencyMapType &latency_map,
                                       const RankCommMapType &rank_comm) {
  std::vector<int> mapping(size);
  std::iota(mapping.begin(), mapping.end(), 0);

  double best_cost = compute_cost(mapping, latency_map, rank_comm, size);
  bool improved = true;

  int rank{};
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::cout << "start 2opt" << std::endl;
  // 2 opt
  while (improved) {
    improved = false;
    for (int i = 0; i < size; ++i) {
      for (int j = i + 1; j < size; ++j) {
        double new_cost =
            update_cost(mapping, latency_map, rank_comm, size, i, j, best_cost);
        std::swap(mapping[i], mapping[j]);
        if (new_cost < best_cost) {
          if (rank == 0)
            std::cout << "improved: " << best_cost << " to " << new_cost
                      << std::endl;
          best_cost = new_cost;
          improved = true;
        } else {
          std::swap(mapping[i], mapping[j]);
        }
      }
    }
  }
  std::cout << "finished 2opt" << std::endl;

  return mapping;
}

RankCommMapType read_communication_profile(const std::string &filename,
                                           int world_size) {
  RankCommMapType rank_comm(0, PairHash{});

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

// todo: the problem with 2opt is it makes miniscule improvements
// then doesn't converge
// so, stop after improvements become small
// or granularise latencies
// I'm looking at replacing 2opt anyway
LatencyMapType compute_latency_map() {
  int world_rank, world_size;
  PMPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const int PING_PONGS = 50;
  const int TAG = 999;
  const int MSG_SIZE = 8;
  std::vector<char> message(MSG_SIZE, 'a');

  for (int i = 0; i < world_size; ++i) {
    for (int j = i + 1; j < world_size; ++j) {

      double latency = 0.0;

      PMPI_Barrier(MPI_COMM_WORLD);
      if (world_rank == i || world_rank == j) {
        int other = (world_rank == i) ? j : i;

        auto start = std::chrono::high_resolution_clock::now();

        for (int k = 0; k < PING_PONGS; ++k) {
          if (world_rank < other) {
            PMPI_Send(message.data(), MSG_SIZE, MPI_CHAR, other, TAG,
                      MPI_COMM_WORLD);
            PMPI_Recv(message.data(), MSG_SIZE, MPI_CHAR, other, TAG,
                      MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          } else {
            PMPI_Recv(message.data(), MSG_SIZE, MPI_CHAR, other, TAG,
                      MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            PMPI_Send(message.data(), MSG_SIZE, MPI_CHAR, other, TAG,
                      MPI_COMM_WORLD);
          }
        }

        auto end = std::chrono::high_resolution_clock::now();
        latency = std::chrono::duration<double>(end - start).count() /
                  (2.0 * PING_PONGS);
      }

      PMPI_Bcast(&latency, 1, MPI_DOUBLE, i, MPI_COMM_WORLD);

      latency_map[{i, j}] = latency;
      latency_map[{j, i}] = latency;
    }
  }
  return latency_map;
}

// todo: replace with hwloc
int get_socket_id() {
  int cpu_id = sched_getcpu();
  char path[256];
  snprintf(path, sizeof(path),
           "/sys/devices/system/cpu/cpu%d/topology/physical_package_id",
           cpu_id);
  FILE *file = fopen(path, "r");
  if (!file)
    throw "couldn't find socket id";
  int socket_id;
  fscanf(file, "%d", &socket_id);
  fclose(file);
  return socket_id;
}

//
LatencyMapType compute_latency_map_tree() {
  int world_rank, world_size;
  PMPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &world_size);

  MPI_Comm node_comm;
  PMPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank,
                       MPI_INFO_NULL, &node_comm);

  int node_comm_size{};
  PMPI_Comm_size(node_comm, &node_comm_size);
  std::vector<int> node_ranks(node_comm_size);
  PMPI_Allgather(&world_rank, 1, MPI_INT, node_ranks.data(), 1, MPI_INT,
                 node_comm);

  int socket_id = get_socket_id();
  MPI_Comm socket_comm;
  MPI_Comm_split(node_comm, socket_id, 0, &socket_comm);

  int socket_comm_size{};
  PMPI_Comm_size(socket_comm, &socket_comm_size);
  std::vector<int> socket_ranks(socket_comm_size);
  PMPI_Allgather(&world_rank, 1, MPI_INT, socket_ranks.data(), 1, MPI_INT,
                 socket_comm);

  // Arbitrary rank
  for (int i = 0; i < world_size; ++i) {
    latency_map[{i, world_rank}] = 4;
    latency_map[{world_rank, i}] = 4;
  }

  // Same node
  for (int rank : node_ranks) {
    latency_map[{rank, world_rank}] = 2;
    latency_map[{world_rank, rank}] = 2;
  }

  // Same socket
  for (int rank : socket_ranks) {
    latency_map[{rank, world_rank}] = 1;
    latency_map[{world_rank, rank}] = 1;
  }

  // todo: does it really make sense to 1) store the latency map explicitly
  // 2) make every rank aware of the latency map
  for (int i = 0; i < world_size; ++i) {
    for (int j = i + 1; j < world_size; ++j) {
      double latency;
      if (world_rank == i) {
        latency = latency_map[{i, j}];
      }
      PMPI_Bcast(&latency, 1, MPI_DOUBLE, i, MPI_COMM_WORLD);
      latency_map[{i, j}] = latency;
      latency_map[{j, i}] = latency;
    }
  }

  return latency_map;
}

// I'm using Rankparticipant to store strategies for computing rank placements
// I probably want to rename this
template <typename Derived> struct RankParticipant {
  int get_new_rank_c();
  int get_new_rank_fortran();
};

// todo: do I even need/want crtp
struct Pt2PtRankParticipant : RankParticipant<Pt2PtRankParticipant> {
  /// requires PMPI_init called
  int get_new_rank_c() {
    int world_rank, world_size;
    PMPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &world_size);
    global_rank = world_rank;
    latency_map = compute_latency_map();

    auto rank_comm = read_communication_profile(
        "sampi_communication_profile.txt", world_size);
    rank_map = compute_rank_map(world_size, latency_map, rank_comm);
    int new_rank = rank_map[world_rank];
    return new_rank;
  }

  int get_new_rank_fortran() {
    // todo: check errors
    int world_rank, world_size, ierr;
    MPI_Fint f_comm_world = MPI_Comm_c2f(MPI_COMM_WORLD);
    pmpi_comm_rank_(&f_comm_world, &world_rank, &ierr);
    pmpi_comm_size_(&f_comm_world, &world_size, &ierr);

    global_rank = world_rank;
    latency_map = compute_latency_map();

    auto rank_comm = read_communication_profile(
        "sampi_communication_profile.txt", world_size);
    rank_map = compute_rank_map(world_size, latency_map, rank_comm);
    int new_rank = rank_map[world_rank];
    return new_rank;
  }
};

// todo: same crtp concern as above
// todo: probably want to refactor, the two strategies only differ in how they
// calculate the cost matrix So might want to use composition one component gets
// the cost matrix one component computes the rank placement
struct TreeMapRankParticipant : RankParticipant<Pt2PtRankParticipant> {
  /// requires PMPI_init called
  int get_new_rank_c() {
    int world_rank, world_size;
    PMPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &world_size);
    global_rank = world_rank;
    latency_map = compute_latency_map_tree();

    auto rank_comm = read_communication_profile(
        "sampi_communication_profile.txt", world_size);
    rank_map = compute_rank_map_tree(world_size, latency_map, rank_comm);
    int new_rank = rank_map[world_rank];
    return new_rank;
  }

  int get_new_rank_fortran() {
    // todo: check errors
    int world_rank, world_size, ierr;
    MPI_Fint f_comm_world = MPI_Comm_c2f(MPI_COMM_WORLD);
    pmpi_comm_rank_(&f_comm_world, &world_rank, &ierr);
    pmpi_comm_size_(&f_comm_world, &world_size, &ierr);

    global_rank = world_rank;
    latency_map = compute_latency_map_tree();

    auto rank_comm = read_communication_profile(
        "sampi_communication_profile.txt", world_size);
    rank_map = compute_rank_map_tree(world_size, latency_map, rank_comm);
    int new_rank = rank_map[world_rank];
    return new_rank;
  }
};

static auto rank_participant = TreeMapRankParticipant{};

int MPI_Init(int *argc, char ***argv) {
  std::cout << "[ALARM] MPI_Init called" << std::endl;
  int return_value = PMPI_Init(argc, argv);
  // todo: check return value
  int new_rank = rank_participant.get_new_rank_c();
  int err = PMPI_Comm_split(MPI_COMM_WORLD, 0, new_rank, &reorder_comm_world);
  // todo: check err value
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

int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
             int tag, MPI_Comm comm) {
  return PMPI_Send(buf, count, datatype, dest, tag, reorder(comm));
}

int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
             MPI_Comm comm, MPI_Status *status) {
  return PMPI_Recv(buf, count, datatype, source, tag, reorder(comm), status);
}

int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 int dest, int sendtag, void *recvbuf, int recvcount,
                 MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm,
                 MPI_Status *status) {

  return PMPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf,
                       recvcount, recvtype, source, recvtag, reorder(comm),
                       status);
}

int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm, MPI_Request *request) {
  return PMPI_Isend(buf, count, datatype, dest, tag, reorder(comm), request);
}

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
              MPI_Comm comm, MPI_Request *request) {
  return PMPI_Irecv(buf, count, datatype, source, tag, reorder(comm), request);
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
  return PMPI_Bcast(buffer, count, datatype, root, reorder(comm));
}

int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
               MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
  return PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root,
                     reorder(comm));
}

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, reorder(comm));
}

int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                MPI_Comm comm) {
  return PMPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                      recvtype, root, reorder(comm));
}

int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
               MPI_Comm comm) {
  return PMPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                     root, reorder(comm));
}

int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  MPI_Comm comm) {
  return PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                        recvtype, reorder(comm));
}

int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 MPI_Comm comm) {
  return PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                       recvtype, reorder(comm));
}

int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, const int recvcounts[], const int displs[],
                MPI_Datatype recvtype, int root, MPI_Comm comm) {
  return PMPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs,
                      recvtype, root, reorder(comm));
}

int MPI_Scatterv(const void *sendbuf, const int sendcounts[],
                 const int displs[], MPI_Datatype sendtype, void *recvbuf,
                 int recvcount, MPI_Datatype recvtype, int root,
                 MPI_Comm comm) {
  return PMPI_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf,
                       recvcount, recvtype, root, reorder(comm));
}

int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, const int recvcounts[], const int displs[],
                   MPI_Datatype recvtype, MPI_Comm comm) {
  return PMPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts,
                         displs, recvtype, reorder(comm));
}

int MPI_Alltoallv(const void *sendbuf, const int sendcounts[],
                  const int sdispls[], MPI_Datatype sendtype, void *recvbuf,
                  const int recvcounts[], const int rdispls[],
                  MPI_Datatype recvtype, MPI_Comm comm) {
  return PMPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf,
                        recvcounts, rdispls, recvtype, reorder(comm));
}

int MPI_Barrier(MPI_Comm comm) { return PMPI_Barrier(reorder(comm)); }

int MPI_Comm_rank(MPI_Comm comm, int *rank) {
  int ret = PMPI_Comm_rank(reorder(comm), rank);
  return ret;
}

int MPI_Comm_size(MPI_Comm comm, int *size) {
  int ret = PMPI_Comm_size(reorder(comm), size);
  return ret;
}

int MPI_Cart_create(MPI_Comm comm_old, int ndims, const int dims[],
                    const int periods[], int reorder, MPI_Comm *comm_cart) {
  return PMPI_Cart_create(reorder(comm_old), ndims, dims, periods, reorder,
                          comm_cart);
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
  if (*ierror != MPI_SUCCESS)
    return;

  int new_rank = rank_participant.get_new_rank_fortran();
  MPI_Fint f_new_comm;
  int colour = 0;
  int ierr;
  MPI_Fint f_comm_world = MPI_Comm_c2f(MPI_COMM_WORLD);
  pmpi_comm_split_(&f_comm_world, &colour, &new_rank, &f_new_comm, &ierr);
  // todo: check error value
  reorder_comm_world = MPI_Comm_f2c(f_new_comm);
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

void mpi_send_(void *buf, int *count, MPI_Fint *datatype, int *dest, int *tag,
               MPI_Fint *comm, int *ierror) {
  MPI_Fint reordered_comm = reorder_comm_f(*comm);
  pmpi_send_(buf, count, datatype, dest, tag, &reordered_comm, ierror);
}

void mpi_recv_(void *buf, int *count, MPI_Fint *datatype, int *source, int *tag,
               MPI_Fint *comm, MPI_Fint *status_f, int *ierror) {
  MPI_Fint reordered_comm = reorder_comm_f(*comm);
  pmpi_recv_(buf, count, datatype, source, tag, &reordered_comm, status_f,
             ierror);
}

void mpi_sendrecv_(void *sendbuf, int *sendcount, MPI_Fint *sendtype, int *dest,
                   int *sendtag, void *recvbuf, int *recvcount,
                   MPI_Fint *recvtype, int *source, int *recvtag,
                   MPI_Fint *comm, MPI_Fint *status_f, int *ierror) {
  MPI_Fint reordered_comm = reorder_comm_f(*comm);
  pmpi_sendrecv_(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf,
                 recvcount, recvtype, source, recvtag, &reordered_comm,
                 status_f, ierror);
}

void mpi_isend_(void *buf, int *count, MPI_Fint *datatype, int *dest, int *tag,
                MPI_Fint *comm, MPI_Fint *request_f, int *ierror) {
  MPI_Fint reordered_comm = reorder_comm_f(*comm);
  pmpi_isend_(buf, count, datatype, dest, tag, &reordered_comm, request_f,
              ierror);
}

void mpi_irecv_(void *buf, int *count, MPI_Fint *datatype, int *source,
                int *tag, MPI_Fint *comm, MPI_Fint *request_f, int *ierror) {
  MPI_Fint reordered_comm = reorder_comm_f(*comm);
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
  MPI_Fint reordered_comm = reorder_comm_f(*comm);
  pmpi_bcast_(buffer, count, datatype, root, &reordered_comm, ierror);
}

void mpi_reduce_(const void *sendbuf, void *recvbuf, int *count,
                 MPI_Fint *datatype, MPI_Fint *op, int *root, MPI_Fint *comm,
                 int *ierror) {
  MPI_Fint reordered_comm = reorder_comm_f(*comm);
  pmpi_reduce_(sendbuf, recvbuf, count, datatype, op, root, &reordered_comm,
               ierror);
}

void mpi_allreduce_(const void *sendbuf, void *recvbuf, int *count,
                    MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm,
                    int *ierror) {
  MPI_Fint reordered_comm = reorder_comm_f(*comm);
  pmpi_allreduce_(sendbuf, recvbuf, count, datatype, op, &reordered_comm,
                  ierror);
}

void mpi_scatter_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype,
                  void *recvbuf, int *recvcount, MPI_Fint *recvtype, int *root,
                  MPI_Fint *comm, int *ierror) {
  MPI_Fint reordered_comm = reorder_comm_f(*comm);
  pmpi_scatter_(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                root, &reordered_comm, ierror);
}

void mpi_gather_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype,
                 void *recvbuf, int *recvcount, MPI_Fint *recvtype, int *root,
                 MPI_Fint *comm, int *ierror) {
  MPI_Fint reordered_comm = reorder_comm_f(*comm);
  pmpi_gather_(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
               &reordered_comm, ierror);
}

void mpi_allgather_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype,
                    void *recvbuf, int *recvcount, MPI_Fint *recvtype,
                    MPI_Fint *comm, int *ierror) {
  MPI_Fint reordered_comm = reorder_comm_f(*comm);
  pmpi_allgather_(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                  &reordered_comm, ierror);
}

void mpi_alltoall_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype,
                   void *recvbuf, int *recvcount, MPI_Fint *recvtype,
                   MPI_Fint *comm, int *ierror) {
  MPI_Fint reordered_comm = reorder_comm_f(*comm);
  pmpi_alltoall_(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                 &reordered_comm, ierror);
}

void mpi_gatherv_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype,
                  void *recvbuf, int *recvcounts, int *displs,
                  MPI_Fint *recvtype, int *root, MPI_Fint *comm, int *ierror) {
  MPI_Fint reordered_comm = reorder_comm_f(*comm);
  pmpi_gatherv_(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs,
                recvtype, root, &reordered_comm, ierror);
}

void mpi_scatterv_(const void *sendbuf, int *sendcounts, int *displs,
                   MPI_Fint *sendtype, void *recvbuf, int *recvcount,
                   MPI_Fint *recvtype, int *root, MPI_Fint *comm, int *ierror) {
  MPI_Fint reordered_comm = reorder_comm_f(*comm);
  pmpi_scatterv_(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount,
                 recvtype, root, &reordered_comm, ierror);
}

void mpi_allgatherv_(const void *sendbuf, int *sendcount, MPI_Fint *sendtype,
                     void *recvbuf, int *recvcounts, int *displs,
                     MPI_Fint *recvtype, MPI_Fint *comm, int *ierror) {
  MPI_Fint reordered_comm = reorder_comm_f(*comm);
  pmpi_allgatherv_(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs,
                   recvtype, &reordered_comm, ierror);
}

void mpi_alltoallv_(const void *sendbuf, int *sendcounts, int *sdispls,
                    MPI_Fint *sendtype, void *recvbuf, int *recvcounts,
                    int *rdispls, MPI_Fint *recvtype, MPI_Fint *comm,
                    int *ierror) {
  MPI_Fint reordered_comm = reorder_comm_f(*comm);
  pmpi_alltoallv_(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts,
                  rdispls, recvtype, &reordered_comm, ierror);
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

void mpi_cart_create_(MPI_Fint *comm_old, int *ndims, int *dims, int *periods,
                      int *reorder_f, MPI_Fint *comm_cart_f, int *ierror) {
  MPI_Fint reordered_comm = reorder_comm_f(*comm_old);
  pmpi_cart_create_(&reordered_comm, ndims, dims, periods, reorder_f,
                    comm_cart_f, ierror);
}

void mpi_comm_dup_(MPI_Fint *comm, MPI_Fint *newcomm_f, int *ierror) {
  MPI_Fint reordered_comm_old = reorder_comm_f(*comm);
  pmpi_comm_dup_(&reordered_comm_old, newcomm_f, ierror);
}

void mpi_comm_split_(MPI_Fint *comm, int *color, int *key, MPI_Fint *newcomm_f,
                     int *ierror) {
  MPI_Fint reordered_comm_old = reorder_comm_f(*comm);
  pmpi_comm_split_(&reordered_comm_old, color, key, newcomm_f, ierror);
}
}
