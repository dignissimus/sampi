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
  if (!config_)
    return;

  auto it = communicator_participants_.find(comm_name);
  if (it != communicator_participants_.end() &&
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
  auto it = communicator_participants_.find(comm_name);
  if (it != communicator_participants_.end() &&
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

  if (rank == 0) {
    std::vector<std::vector<unsigned long long int>> all_rank_communication(
        world_size, std::vector<unsigned long long int>(world_size, 0));

    for (int i = 0; i < world_size; ++i) {
      all_rank_communication[0][i] = partial_rank_communication_[{0, i}];
      all_rank_communication[i][0] = partial_rank_communication_[{0, i}];
    }

    for (int i = 1; i < world_size; ++i) {
      std::vector<unsigned long long int> rank_data(world_size, 0);
      PMPI_Recv(rank_data.data(), world_size, MPI_LONG_LONG, i, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      for (int j = 0; j < world_size; ++j) {
        all_rank_communication[i][j] += rank_data[j];
        all_rank_communication[j][i] += rank_data[j];
      }
    }
    write_rank_communication_to_file(all_rank_communication);
  } else {
    std::vector<unsigned long long int> send_to_vector(world_size, 0);
    for (int i = 0; i < world_size; ++i) {
      send_to_vector[i] = partial_rank_communication_[{rank, i}];
    }
    PMPI_Send(send_to_vector.data(), world_size, MPI_LONG_LONG, 0, 0,
              MPI_COMM_WORLD);
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

  if (rank == 0) {
    std::vector<std::vector<unsigned long long int>> all_rank_communication(
        world_size, std::vector<unsigned long long int>(world_size, 0));

    for (int i = 0; i < world_size; ++i) {
      all_rank_communication[0][i] = partial_rank_communication_[{0, i}];
      all_rank_communication[i][0] = partial_rank_communication_[{0, i}];
    }

    for (int i = 1; i < world_size; ++i) {
      std::vector<unsigned long long int> rank_data(world_size, 0);
      int count = world_size;
      int source = i;
      int tag = 0;
      pmpi_recv_(rank_data.data(), &count, &f_long_long, &source, &tag,
                 &f_comm_world, (MPI_Fint *)MPI_F_STATUS_IGNORE, &f_ierr);

      for (int j = 0; j < world_size; ++j) {
        all_rank_communication[i][j] += rank_data[j];
        all_rank_communication[j][i] += rank_data[j];
      }
    }
    write_rank_communication_to_file(all_rank_communication);
  } else {
    std::vector<unsigned long long int> send_to_vector(world_size, 0);
    for (int i = 0; i < world_size; ++i) {
      send_to_vector[i] = partial_rank_communication_[{rank, i}];
    }
    int count = world_size;
    int dest = 0;
    int tag = 0;
    pmpi_send_(send_to_vector.data(), &count, &f_long_long, &dest, &tag,
               &f_comm_world, &f_ierr);
  }
}

void Profiler::write_rank_communication_to_file(
    const std::vector<std::vector<unsigned long long int>>
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
