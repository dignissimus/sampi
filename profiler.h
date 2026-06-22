#ifndef PROFILER_H
#define PROFILER_H

#include "fortran_headers.h"
#include <fstream>
#include <map>
#include <mpi.h>
#include <optional>
#include <string>
#include <vector>

class Profiler {
public:
  struct MpiConfig {
    int global_rank;
    int global_world_size;
  };

  static Profiler &instance() { return instance_; }

  void initialise(int rank, int size, const std::string &world_name);

  bool is_initialised() const;

  std::optional<int> get_global_rank() const;

  std::optional<int> get_global_world_size() const;

  std::string
  generate_and_track_communicator(const std::string &original_name,
                                  const std::string &type_label, int suffix,
                                  const std::vector<int> &world_ranks);

  Profiler(const Profiler &) = delete;
  Profiler &operator=(const Profiler &) = delete;

  void record_p2p_send(int dest_communicator_rank,
                       const std::string &comm_name);

  void record_collective_broadcast(const std::string &comm_name);

  int translate_to_global_rank(const std::string &comm_name,
                               int communicator_rank);

  void dump_profile();

  void dump_profile_fortran();

private:
  Profiler() = default;
  static Profiler instance_;

  std::optional<MpiConfig> config_;

  // TODO: Can replace with just a vector since I only record sends from this
  // process
  std::map<std::pair<int, int>, long long int>
      partial_rank_communication_;
  std::map<std::string, std::vector<int>> communicator_participants_;
  std::map<std::string, int> split_count_;

  void write_rank_communication_to_file(
      const std::vector<std::vector<long long int>>
          &all_rank_communication);
};

#endif // PROFILER_H
