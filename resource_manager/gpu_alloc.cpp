#include "resource_manager.h"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>
#include <mpi.h>

std::string build_ssh_command(const std::string& hostname, const std::vector<int>& gpu_ids, const std::vector<std::string>& user_command) {
    std::stringstream visible_devices;
    for (size_t i = 0; i < gpu_ids.size(); i++) {
        visible_devices << gpu_ids[i] << (i == gpu_ids.size() - 1 ? "" : ",");
    }

    std::stringstream ssh_cmd;
    ssh_cmd << "ssh " << hostname << " \"";
    ssh_cmd << "CUDA_VISIBLE_DEVICES=" << visible_devices.str() << " ";
    for (const auto& part : user_command) {
        ssh_cmd << part << " ";
    }
    ssh_cmd << "\"";
    return ssh_cmd.str();
}

int main(int argc, char** argv) {
    if (argc < 4 || std::string(argv[1]) != "--gpus") {
        std::cerr << "Usage: " << argv[0] << " --gpus <N> -- <your_command_and_args>" << std::endl;
        std::cerr << "Example: " << argv[0] << "--gpus 2 -- torchrun my_script.py --epochs 10" << std::endl;
    }

    int num_gpus;
    try {
        num_gpus = std::stoi(argv[2]);
        if (num_gpus < 1) {
            throw std::runtime_error("Invalid number of GPUs");
        }
    } catch (const std::exception& e) {
        std::cerr << "Invalid number of GPUs specified: " << argv[2] << std::endl;
        return 1;
    }

    std::vector<std::string> user_command;
    for (int i = 4; i < argc; i++) {
        user_command.emplace_back(argv[i]);
    }

    MPI_Init(&argc, &argv);

    // REQUEST ALLOCATION
    std::cout << "[Client] Requesting " << num_gpus << " GPUs from master" << std::endl;
    MPI_Send(&num_gpus, 1, MPI_INT, 0, TAG_REQUEST_ALLOCATION, MPI_COMM_WORLD);

    // WAIT FOR RESPONSE
    MPI_Status status;
    MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    if (status.MPI_TAG == TAG_REJECT_ALLOCATION) {
        int sig;
        MPI_Recv(&sig, 1, MPI_INT, 0, TAG_REJECT_ALLOCATION, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cerr << "[Client] Allocation request rejected. Not enough free resources." << std::endl;
        MPI_Finalize();
        return 1;
    }

    int allocation_id;
    char hostname[256];
    std::vector<int> gpu_ids(num_gpus);

    MPI_Recv(&allocation_id, 1, MPI_INT, 0, TAG_REQUEST_ALLOCATION, MPI_COMM_WORLD, &status);
    MPI_Recv(hostname, 256, MPI_CHAR, 0, TAG_GRANT_ALLOCATION, MPI_COMM_WORLD, &status);
    MPI_Recv(gpu_ids.data(), num_gpus, MPI_INT, 0, TAG_GRANT_ALLOCATION, MPI_COMM_WORLD, &status);

    std::cout << "[Client] Allocation " << allocation_id << " granted on node " << hostname << std::endl;

    // REMOTE EXECUTION
    std::string command_to_run = build_ssh_command(hostname, gpu_ids, user_command);
    std::cout << "[Client] Executing command: " << command_to_run << std::endl;
    std::cout << "-------------------- JOB OUTPUT START --------------------" << std::endl;
    int return_code = system(command_to_run.c_str());
    std::cout << "-------------------- JOB OUTPUT END --------------------" << std::endl;

    if (return_code != 0) {
        std::cerr << "[Client] Command execution failed with return code: " << return_code << std::endl;
    }

    // RELEASE ALLOCATION
    std::cout << "[Client] Releasing allocation " << allocation_id << std::endl;
    MPI_Send(&allocation_id, 1, MPI_INT, 0, TAG_RELEASE_ALLOCATION, MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}
