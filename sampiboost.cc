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

struct ReorderConfig {
  std::string profile_filepath = "sampi_communication_profile.txt";

  int ping_pong_count = 50;
  int ping_pong_tag = 999;
  int ping_pong_msg_size = 8;

  double latency_arbitrary = 4.0;
  double latency_same_node = 2.0;
  double latency_same_socket = 1.0;
  double latency_same_core = 0.0;
};

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

class RankReorder {
public:
  static RankReorder &instance() {
    static RankReorder instance;
    return instance;
  }

  RankReorder(const RankReorder &) = delete;
  RankReorder &operator=(const RankReorder &) = delete;

  ReorderConfig &config() { return config_; }

  void set_global_rank(int rank) { global_rank_ = rank; }
  int get_global_rank() const { return global_rank_.value_or(0); }

  void set_latency_map(LatencyMapType map) { latency_map_ = std::move(map); }
  const LatencyMapType &get_latency_map() const { return latency_map_; }

  void set_rank_map(std::vector<int> map) { rank_map_ = std::move(map); }
  const std::vector<int> &get_rank_map() const { return rank_map_; }

  void set_reorder_comm_world(MPI_Comm comm) { reorder_comm_world_ = comm; }
  MPI_Comm get_reorder_comm_world() const { return reorder_comm_world_; }

  MPI_Comm reorder_communicator(MPI_Comm comm) const {
    return (comm == MPI_COMM_WORLD && reorder_comm_world_ != MPI_COMM_NULL)
               ? reorder_comm_world_
               : comm;
  }

  MPI_Fint reorder_communicator_fortran(MPI_Fint f_comm) const {
    MPI_Comm c_comm = MPI_Comm_f2c(f_comm);
    if (c_comm == MPI_COMM_WORLD && reorder_comm_world_ != MPI_COMM_NULL) {
      return MPI_Comm_c2f(reorder_comm_world_);
    }
    return f_comm;
  }

private:
  RankReorder() = default;

  ReorderConfig config_;
  std::optional<int> global_rank_;
  LatencyMapType latency_map_{0, PairHash{}};
  std::vector<int> rank_map_;
  MPI_Comm reorder_comm_world_ = MPI_COMM_NULL;
};

struct PingPongCostModel {
  static LatencyMapType compute_latency() {
    LatencyMapType local_latency_map(0, PairHash{});
    int world_rank, world_size;
    PMPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::vector<char> message(
        RankReorder::instance().config().ping_pong_msg_size, 'a');

    for (int i = 0; i < world_size; ++i) {
      for (int j = i + 1; j < world_size; ++j) {
        double latency = 0.0;
        PMPI_Barrier(MPI_COMM_WORLD);

        if (world_rank == i || world_rank == j) {
          int other = (world_rank == i) ? j : i;
          auto start = std::chrono::high_resolution_clock::now();

          for (int k = 0; k < RankReorder::instance().config().ping_pong_count;
               ++k) {
            if (world_rank < other) {
              PMPI_Send(message.data(),
                        RankReorder::instance().config().ping_pong_msg_size,
                        MPI_CHAR, other,
                        RankReorder::instance().config().ping_pong_tag,
                        MPI_COMM_WORLD);
              PMPI_Recv(message.data(),
                        RankReorder::instance().config().ping_pong_msg_size,
                        MPI_CHAR, other,
                        RankReorder::instance().config().ping_pong_tag,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
              PMPI_Recv(message.data(),
                        RankReorder::instance().config().ping_pong_msg_size,
                        MPI_CHAR, other,
                        RankReorder::instance().config().ping_pong_tag,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              PMPI_Send(message.data(),
                        RankReorder::instance().config().ping_pong_msg_size,
                        MPI_CHAR, other,
                        RankReorder::instance().config().ping_pong_tag,
                        MPI_COMM_WORLD);
            }
          }
          auto end = std::chrono::high_resolution_clock::now();
          latency = std::chrono::duration<double>(end - start).count() /
                    (2.0 * RankReorder::instance().config().ping_pong_count);
        }

        PMPI_Bcast(&latency, 1, MPI_DOUBLE, i, MPI_COMM_WORLD);
        local_latency_map[{i, j}] = latency;
        local_latency_map[{j, i}] = latency;
      }
    }
    return local_latency_map;
  }
};
int get_socket_id() {
  int cpu_id = sched_getcpu();
  std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu_id) +
                     "/topology/physical_package_id";

  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error(
        "[SAMPI] ERROR: Could not open socket topology file for CPU " +
        std::to_string(cpu_id));
  }

  int socket_id;
  if (!(file >> socket_id)) {
    throw std::runtime_error("[SAMPI] ERROR: Could not parse socket ID from " +
                             path);
  }

  return socket_id;
}
struct TreeCostModel {

  static LatencyMapType compute_latency() {
    LatencyMapType local_latency_map(0, PairHash{});
    int world_rank, world_size;
    PMPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Comm node_comm;
    PMPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank,
                         MPI_INFO_NULL, &node_comm);

    int node_comm_size;
    PMPI_Comm_size(node_comm, &node_comm_size);
    std::vector<int> node_ranks(node_comm_size);
    PMPI_Allgather(&world_rank, 1, MPI_INT, node_ranks.data(), 1, MPI_INT,
                   node_comm);

    int socket_id = get_socket_id();
    MPI_Comm socket_comm;
    PMPI_Comm_split(node_comm, socket_id, 0, &socket_comm);

    int socket_comm_size;
    PMPI_Comm_size(socket_comm, &socket_comm_size);
    std::vector<int> socket_ranks(socket_comm_size);
    PMPI_Allgather(&world_rank, 1, MPI_INT, socket_ranks.data(), 1, MPI_INT,
                   socket_comm);

    for (int i = 0; i < world_size; ++i) {
      local_latency_map[{i, world_rank}] =
          RankReorder::instance().config().latency_arbitrary;
      local_latency_map[{world_rank, i}] =
          RankReorder::instance().config().latency_arbitrary;
    }
    for (int rank : node_ranks) {
      local_latency_map[{rank, world_rank}] =
          RankReorder::instance().config().latency_same_node;
      local_latency_map[{world_rank, rank}] =
          RankReorder::instance().config().latency_same_node;
    }
    for (int rank : socket_ranks) {
      local_latency_map[{rank, world_rank}] =
          RankReorder::instance().config().latency_same_socket;
      local_latency_map[{world_rank, rank}] =
          RankReorder::instance().config().latency_same_socket;
    }
    local_latency_map[{world_rank, world_rank}] =
        RankReorder::instance().config().latency_same_core;

    for (int i = 0; i < world_size; ++i) {
      for (int j = i; j < world_size; ++j) {
        double latency = (world_rank == i) ? local_latency_map[{i, j}] : 0.0;
        PMPI_Bcast(&latency, 1, MPI_DOUBLE, i, MPI_COMM_WORLD);
        local_latency_map[{i, j}] = latency;
        local_latency_map[{j, i}] = latency;
      }
    }
    return local_latency_map;
  }
};

template <typename CostModel> class TwoOptOptimiser {
private:
  static int get_comm_volume(const RankCommMapType &rank_comm, int src,
                             int dst) {
    auto it = rank_comm.find({src, dst});
    return it != rank_comm.end() ? it->second : 0;
  }

  static double compute_cost(const std::vector<int> &mapping,
                             const LatencyMapType &latency_map,
                             const RankCommMapType &rank_comm, int size) {
    double cost = 0.0;
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < i; ++j) {
        int pi = mapping[i];
        int pj = mapping[j];

        int forwards_comm = 0, backwards_comm = 0;
        auto it_fwd = rank_comm.find({pi, pj});
        if (it_fwd != rank_comm.end())
          forwards_comm = it_fwd->second;

        auto it_bwd = rank_comm.find({pj, pi});
        if (it_bwd != rank_comm.end())
          backwards_comm = it_bwd->second;

        auto total_comm = forwards_comm + backwards_comm;
        auto forwards_latency = latency_map.at({i, j});
        auto backwards_latency = latency_map.at({j, i});
        auto total_latency = forwards_latency + backwards_latency;

        cost += total_comm * total_latency;
      }
    }
    return cost;
  }

  static double update_cost(const std::vector<int> &mapping,
                            const LatencyMapType &latency_map,
                            const RankCommMapType &rank_comm, int size, int si,
                            int sj, double original_cost) {
    double cost_difference = 0.0;
    int pi = mapping[si];
    int pj = mapping[sj];

    for (int sx = 0; sx < size; ++sx) {
      if (sx == si || sx == sj)
        continue;

      int px = mapping[sx];

      auto total_comm_i = get_comm_volume(rank_comm, px, pi) +
                          get_comm_volume(rank_comm, pi, px);
      auto total_comm_j = get_comm_volume(rank_comm, px, pj) +
                          get_comm_volume(rank_comm, pj, px);

      auto total_latency_i =
          latency_map.at({sx, si}) + latency_map.at({si, sx});
      auto total_latency_j =
          latency_map.at({sx, sj}) + latency_map.at({sj, sx});

      cost_difference += (total_comm_j - total_comm_i) * total_latency_i;
      cost_difference += (total_comm_i - total_comm_j) * total_latency_j;
    }
    return original_cost + cost_difference;
  }

public:
  static std::vector<int> optimise(int world_size,
                                   const RankCommMapType &rank_comm,
                                   LatencyMapType &out_latency_map) {

    out_latency_map = CostModel::compute_latency();

    std::vector<int> mapping(world_size);
    std::iota(mapping.begin(), mapping.end(), 0);

    double best_cost =
        compute_cost(mapping, out_latency_map, rank_comm, world_size);
    bool improved = true;

    while (improved) {
      improved = false;
      for (int i = 0; i < world_size; ++i) {
        for (int j = i + 1; j < world_size; ++j) {
          double new_cost = update_cost(mapping, out_latency_map, rank_comm,
                                        world_size, i, j, best_cost);

          if (new_cost < best_cost) {
            best_cost = new_cost;
            std::swap(mapping[i], mapping[j]);
            improved = true;
          }
        }
      }
    }
    return mapping;
  }
};

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
  return rank_comm;
}

static void optimise_placement(int world_rank, int world_size) {
  RankReorder::instance().set_global_rank(world_rank);

  auto rank_comm = read_communication_profile(
      RankReorder::instance().config().profile_filepath, world_size);

  LatencyMapType temp_latency_map;

  std::vector<int> temp_rank_map = TwoOptOptimiser<TreeCostModel>::optimise(
      world_size, rank_comm, temp_latency_map);

  RankReorder::instance().set_latency_map(std::move(temp_latency_map));
  RankReorder::instance().set_rank_map(std::move(temp_rank_map));
}

// TODO: Read error value
int MPI_Init(int *argc, char ***argv) {
  int return_value = PMPI_Init(argc, argv);
  if (return_value != MPI_SUCCESS)
    return return_value;

  int world_rank, world_size;
  PMPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &world_size);

  optimise_placement(world_rank, world_size);

  int new_rank = RankReorder::instance().get_rank_map()[world_rank];
  MPI_Comm new_comm;
  int err = PMPI_Comm_split(MPI_COMM_WORLD, 0, new_rank, &new_comm);
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

  optimise_placement(world_rank, world_size);

  int new_rank = RankReorder::instance().get_rank_map()[world_rank];
  MPI_Comm new_comm;
  int err = PMPI_Comm_split(MPI_COMM_WORLD, 0, new_rank, &new_comm);

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

  optimise_placement(world_rank, world_size);

  int new_rank = RankReorder::instance().get_rank_map()[world_rank];
  MPI_Fint f_new_comm;
  int colour = 0;

  pmpi_comm_split_(&f_comm_world, &colour, &new_rank, &f_new_comm, ierror);

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

  optimise_placement(world_rank, world_size);

  int new_rank = RankReorder::instance().get_rank_map()[world_rank];
  MPI_Fint f_new_comm;
  int colour = 0;

  pmpi_comm_split_(&f_comm_world, &colour, &new_rank, &f_new_comm, ierror);

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
