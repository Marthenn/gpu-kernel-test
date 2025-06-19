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
#include <fstream>
#include <wordexp.h> // Needed for tilde (~) expansion

// Helper to build the final ssh command
std::string build_ssh_command(
    const std::string& hostname,
    const std::vector<int>& gpu_ids,
    const std::vector<std::string>& user_command) {

    std::stringstream visible_devices;
    for (size_t i = 0; i < gpu_ids.size(); ++i) {
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

int main(int argc, char* argv[]) {
    if (argc < 4 || std::string(argv[1]) != "--gpus") {
        std::cerr << "Usage: " << argv[0] << " --gpus <N> -- <your_command_and_args>" << std::endl;
        std::cerr << "Example: " << argv[0] << " --gpus 2 -- torchrun my_script.py --epochs 10" << std::endl;
        return 1;
    }

    int num_gpus;
    try {
        num_gpus = std::stoi(argv[2]);
    } catch (const std::exception& e) {
        std::cerr << "Invalid number of GPUs requested." << std::endl;
        return 1;
    }

    std::vector<std::string> user_command;
    for (int i = 4; i < argc; ++i) {
        user_command.push_back(argv[i]);
    }

    MPI_Init(&argc, &argv);

    char port_name[MPI_MAX_PORT_NAME];
    MPI_Comm server_comm;

    // Step 1: Read the server's port name from a well-known file
    std::cout << "[Client] Reading master daemon port from file..." << std::endl;
    std::string port_file_path_str;
    wordexp_t p;
    // Use wordexp to handle the '~' in the path correctly
    if (wordexp("~/.gpu_manager.port", &p, 0) == 0) {
        port_file_path_str = p.we_wordv[0];
        wordfree(&p);
    } else {
        std::cerr << "[Client] Error: Could not expand home directory path '~/.gpu_manager.port'." << std::endl;
        MPI_Finalize();
        return 1;
    }

    std::ifstream infile(port_file_path_str);
    if (!infile.is_open()) {
        std::cerr << "[Client] Error: Could not open port file '" << port_file_path_str << "'." << std::endl;
        std::cerr << "Is the resource_manager_daemon running and has it created the port file?" << std::endl;
        MPI_Finalize();
        return 1;
    }
    infile >> port_name;
    infile.close();

    // Step 2: Connect to the server using the port name from the file
    std::cout << "[Client] Connecting to master daemon..." << std::endl;
    if (MPI_Comm_connect(port_name, MPI_INFO_NULL, 0, MPI_COMM_SELF, &server_comm) != MPI_SUCCESS) {
        std::cerr << "[Client] Error: Could not connect to the master daemon." << std::endl;
        MPI_Finalize();
        return 1;
    }

    // Step 3: Request Allocation using the new communicator
    std::cout << "[Client] Requesting " << num_gpus << " GPUs from master..." << std::endl;
    MPI_Send(&num_gpus, 1, MPI_INT, 0, TAG_REQUEST_ALLOCATION, server_comm);

    // Step 4: Wait for master's response on the new communicator
    MPI_Status status;
    MPI_Probe(0, MPI_ANY_TAG, server_comm, &status);

    if (status.MPI_TAG == TAG_REJECT_ALLOCATION) {
        int sig;
        MPI_Recv(&sig, 1, MPI_INT, 0, TAG_REJECT_ALLOCATION, server_comm, MPI_STATUS_IGNORE);
        std::cerr << "[Client] Allocation request rejected. Not enough free resources." << std::endl;
    } else {
        int allocation_id;
        char hostname[256];
        std::vector<int> gpu_ids(num_gpus);

        MPI_Recv(&allocation_id, 1, MPI_INT, 0, TAG_GRANT_ALLOCATION, server_comm, MPI_STATUS_IGNORE);
        MPI_Recv(hostname, 256, MPI_CHAR, 0, TAG_GRANT_ALLOCATION, server_comm, MPI_STATUS_IGNORE);
        MPI_Recv(gpu_ids.data(), num_gpus, MPI_INT, 0, TAG_GRANT_ALLOCATION, server_comm, MPI_STATUS_IGNORE);

        std::cout << "[Client] Allocation " << allocation_id << " granted on node " << hostname << std::endl;

        std::string command_to_run = build_ssh_command(hostname, gpu_ids, user_command);
        std::cout << "[Client] Executing: " << command_to_run << std::endl;
        std::cout << "-------------------- JOB OUTPUT START --------------------" << std::endl;
        int return_code = system(command_to_run.c_str());
        std::cout << "-------------------- JOB OUTPUT END --------------------" << std::endl;

        if (return_code != 0) {
            std::cerr << "[Client] User command finished with non-zero exit code: " << return_code << std::endl;
        }

        std::cout << "[Client] Releasing allocation " << allocation_id << "..." << std::endl;
        MPI_Send(&allocation_id, 1, MPI_INT, 0, TAG_RELEASE_ALLOCATION, server_comm);
    }

    // Step 5: Disconnect and finalize
    MPI_Comm_disconnect(&server_comm);
    MPI_Finalize();
    return 0;
}
