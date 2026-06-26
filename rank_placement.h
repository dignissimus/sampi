#ifndef RANK_PLACEMENT_H
#define RANK_PLACEMENT_H

#include "rank_reorder.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// TODO: Probably don't want most of these to be static
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

inline int get_socket_id() {
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
        long long int latency = (world_rank == i) ? local_latency_map[{i, j}] : 0LL;
        PMPI_Bcast(&latency, 1, MPI_LONG_LONG_INT, i, MPI_COMM_WORLD);
        local_latency_map[{i, j}] = latency;
        local_latency_map[{j, i}] = latency;
      }
    }
    return local_latency_map;
  }
};

template <typename CostModel> class TwoOptOptimiser {
private:
  static long long int get_comm_volume(const RankCommMapType &rank_comm, int src,
                             int dst) {
    auto it = rank_comm.find({src, dst});
    return it != rank_comm.end() ? it->second : 0LL;
  }

  static long long int compute_cost(const std::vector<int> &mapping,
                             const LatencyMapType &latency_map,
                             const RankCommMapType &rank_comm, int size) {
    long long int cost = 0LL;
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < i; ++j) {
        int pi = mapping[i];
        int pj = mapping[j];

        long long int forwards_comm = 0, backwards_comm = 0;
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

  static long long int update_cost(const std::vector<int> &mapping,
                            const LatencyMapType &latency_map,
                            const RankCommMapType &rank_comm, int size, int si,
                            int sj, long long int original_cost) {
    long long int cost_difference = 0LL;
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

    long long int best_cost =
        compute_cost(mapping, out_latency_map, rank_comm, world_size);
    bool improved = true;
    int iteration_count = 0;
    int world_rank;
    PMPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    while (improved) {
      improved = false;
      iteration_count++;

      if (world_rank == 0) {
        auto now = std::chrono::system_clock::now();
        auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now).time_since_epoch().count();
        std::cout << "[TwoOpt] Iteration: " << iteration_count 
                  << ", Timestamp: " << now_ms << " ms"
                  << ", Current Cost: " << best_cost << std::endl;
      }

      for (int i = 0; i < world_size; ++i) {
        for (int j = i + 1; j < world_size; ++j) {
          long long int new_cost = update_cost(mapping, out_latency_map, rank_comm,
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

inline RankCommMapType read_communication_profile(const std::string &filename,
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
    long long int value;
    int j = 0;
    while (iss >> value && j < world_size) {
      rank_comm[{i, j}] = value;
      j++;
    }
    i++;
  }
  return rank_comm;
}

inline void optimise_placement(int world_rank, int world_size) {
  RankReorder::instance().set_global_rank(world_rank);

  auto rank_comm = read_communication_profile(
      RankReorder::instance().config().profile_filepath, world_size);

  LatencyMapType temp_latency_map;

  std::vector<int> temp_rank_map = TwoOptOptimiser<TreeCostModel>::optimise(
      world_size, rank_comm, temp_latency_map);

  RankReorder::instance().set_latency_map(std::move(temp_latency_map));
  RankReorder::instance().set_rank_map(std::move(temp_rank_map));
}

#endif // RANK_PLACEMENT_H
