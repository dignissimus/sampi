#ifndef RANK_REORDER_H
#define RANK_REORDER_H

#include <mpi.h>
#include <optional>
#include <string>
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

#endif // RANK_REORDER_H
