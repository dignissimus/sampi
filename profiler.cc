#include "profiler.h"
#include <iostream>
#include <numeric>

Profiler Profiler::instance_;

void Profiler::initialise(int rank, int size, const std::string &world_name) {
  if (config_.has_value())
    return;

  config_ = MpiConfig{rank, size};

  auto &participants = communicator_participants_[world_name];
  participants.resize(size);
  std::iota(participants.begin(), participants.end(), 0);

  std::cout << "[SAMPI] Profiler initialised on rank " << config_->global_rank
            << std::endl;
}

bool Profiler::is_initialised() const { return config_.has_value(); }

std::optional<int> Profiler::get_global_rank() const {
  if (!config_)
    return std::nullopt;
  return config_->global_rank;
}

std::optional<int> Profiler::get_global_world_size() const {
  if (!config_)
    return std::nullopt;
  return config_->global_world_size;
}

std::string Profiler::generate_and_track_communicator(
    const std::string &original_name, const std::string &type_label, int suffix,
    const std::vector<int> &world_ranks) {
  int id = ++split_count_[original_name];

  std::string new_name =
      original_name + " (" + type_label + ") (id=" + std::to_string(id) + ")";
  if (type_label == "split") {
    new_name += " (colour=" + std::to_string(suffix) + ")";
  }

  communicator_participants_[new_name] = world_ranks;

  return new_name;
}

void Profiler::record_p2p_send(int dest_communicator_rank,
                               const std::string &comm_name) {
  if (!config_ || dest_communicator_rank == MPI_PROC_NULL)
    return;

  auto it = communicator_participants_.find(comm_name);
  if (it != communicator_participants_.end() &&
      dest_communicator_rank >= 0 &&
      dest_communicator_rank < static_cast<int>(it->second.size())) {
    int global_dest = it->second[dest_communicator_rank];
    ++partial_rank_communication_[{config_->global_rank, global_dest}];
  }
}

void Profiler::record_collective_broadcast(const std::string &comm_name) {
  if (!config_)
    return;

  auto it = communicator_participants_.find(comm_name);
  if (it != communicator_participants_.end()) {
    for (int global_participant : it->second) {
      ++partial_rank_communication_[{config_->global_rank, global_participant}];
    }
  }
}

int Profiler::translate_to_global_rank(const std::string &comm_name,
                                       int communicator_rank) {
  if (communicator_rank == MPI_PROC_NULL)
    return MPI_PROC_NULL;

  auto it = communicator_participants_.find(comm_name);
  if (it != communicator_participants_.end() &&
      communicator_rank >= 0 &&
      communicator_rank < static_cast<int>(it->second.size())) {
    return it->second[communicator_rank];
  }
  return communicator_rank;
}

void Profiler::dump_profile() {
  if (!config_)
    return;
  int rank = config_->global_rank;
  int world_size = config_->global_world_size;

  std::vector<long long int> local_communication_counts(world_size, 0);
  for (int i = 0; i < world_size; ++i) {
    local_communication_counts[i] = partial_rank_communication_[{rank, i}];
  }

  std::vector<long long int> gathered_communication_counts;
  if (rank == 0) {
    gathered_communication_counts.resize(world_size * world_size, 0);
  }

  PMPI_Gather(local_communication_counts.data(), world_size, MPI_LONG_LONG,
              gathered_communication_counts.data(), world_size, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::vector<std::vector<long long int>> all_rank_communication(
        world_size, std::vector<long long int>(world_size, 0));

    for (int i = 0; i < world_size; ++i) {
      for (int j = 0; j < world_size; ++j) {
        long long int messages_sent = gathered_communication_counts[i * world_size + j];
        all_rank_communication[i][j] += messages_sent;
        if (i != j) {
          all_rank_communication[j][i] += messages_sent;
        }
      }
    }
    write_rank_communication_to_file(all_rank_communication);
  }
}

void Profiler::dump_profile_fortran() {
  if (!config_)
    return;
  int rank = config_->global_rank;
  int world_size = config_->global_world_size;

  MPI_Fint f_comm_world = MPI_Comm_c2f(MPI_COMM_WORLD);
  MPI_Fint f_long_long = MPI_Type_c2f(MPI_LONG_LONG);
  int f_ierr;

  std::vector<long long int> local_communication_counts(world_size, 0);
  for (int i = 0; i < world_size; ++i) {
    local_communication_counts[i] = partial_rank_communication_[{rank, i}];
  }

  std::vector<long long int> gathered_communication_counts;
  if (rank == 0) {
    gathered_communication_counts.resize(world_size * world_size, 0);
  }

  int count = world_size;
  int root = 0;
  pmpi_gather_(local_communication_counts.data(), &count, &f_long_long,
               gathered_communication_counts.data(), &count, &f_long_long, &root,
               &f_comm_world, &f_ierr);

  if (rank == 0) {
    std::vector<std::vector<long long int>> all_rank_communication(
        world_size, std::vector<long long int>(world_size, 0));

    for (int i = 0; i < world_size; ++i) {
      for (int j = 0; j < world_size; ++j) {
        long long int messages_sent = gathered_communication_counts[i * world_size + j];
        all_rank_communication[i][j] += messages_sent;
        if (i != j) {
          all_rank_communication[j][i] += messages_sent;
        }
      }
    }
    write_rank_communication_to_file(all_rank_communication);
  }
}

void Profiler::write_rank_communication_to_file(
    const std::vector<std::vector<long long int>>
        &all_rank_communication) {
  std::ofstream outfile("sampi_communication_profile.txt");
  if (!outfile.is_open()) {
    std::cerr
        << "Error: Could not open output file sampi_communication_profile.txt"
        << std::endl;
    return;
  }

  for (const auto &row : all_rank_communication) {
    for (const auto &number : row) {
      outfile << number << ' ';
    }
    outfile << std::endl;
  }

  outfile.close();
  std::cout << "[PROFILE] Communication data written to "
               "sampi_communication_profile.txt"
            << std::endl;
}
