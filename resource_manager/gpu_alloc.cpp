#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>
#include <fstream>
#include <wordexp.h>
#include <thread>
#include <chrono>

// Helper to expand tilde in path
std::string expand_path(const char* path) {
    wordexp_t p;
    if (wordexp(path, &p, 0) == 0) {
        std::string res = p.we_wordv[0];
        wordfree(&p);
        return res;
    }
    return "";
}

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
        std::cerr << "Example: " << argv[0] << " --gpus 2 -- ./dummy_job" << std::endl;
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

    // --- File-based communication ---
    std::string req_dir = expand_path("~/gpu_requests");
    std::string resp_dir = expand_path("~/gpu_responses");
    std::string release_dir = expand_path("~/gpu_releases");

    // 1. Create a unique request file
    std::string request_id = std::to_string(getpid());
    std::string req_filepath = req_dir + "/" + request_id;
    std::string resp_filepath = resp_dir + "/" + request_id;

    std::ofstream req_file(req_filepath);
    if (!req_file.is_open()) {
        std::cerr << "[Client] Error: Could not create request file in " << req_dir << std::endl;
        return 1;
    }
    req_file << num_gpus << std::endl;
    req_file.close();
    std::cout << "[Client] Submitted request " << request_id << " for " << num_gpus << " GPUs. Waiting for response..." << std::endl;

    // 2. Poll for the response file
    int allocation_id = -1;
    std::string hostname;
    std::vector<int> gpu_ids;

    while (true) {
        std::ifstream resp_file(resp_filepath);
        if (resp_file.is_open()) {
            std::string first_line;
            resp_file >> first_line;
            if (first_line == "REJECTED") {
                std::cerr << "[Client] Allocation request rejected. Not enough free resources." << std::endl;
                resp_file.close();
                std::remove(resp_filepath.c_str());
                return 1;
            }

            // --- Allocation granted ---
            allocation_id = std::stoi(first_line);
            resp_file >> hostname;
            int gpu_id;
            while(resp_file >> gpu_id) {
                gpu_ids.push_back(gpu_id);
            }
            resp_file.close();
            std::remove(resp_filepath.c_str());
            break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::cout << "[Client] Allocation " << allocation_id << " granted on node " << hostname << std::endl;

    // 3. Execute the user's command via SSH
    std::string command_to_run = build_ssh_command(hostname, gpu_ids, user_command);
    std::cout << "[Client] Executing: " << command_to_run << std::endl;
    std::cout << "-------------------- JOB OUTPUT START --------------------" << std::endl;
    int return_code = system(command_to_run.c_str());
    std::cout << "-------------------- JOB OUTPUT END --------------------" << std::endl;

    if (return_code != 0) {
        std::cerr << "[Client] User command finished with non-zero exit code: " << return_code << std::endl;
    }

    // 4. Create a release file to notify the daemon
    std::cout << "[Client] Releasing allocation " << allocation_id << "..." << std::endl;
    std::string release_filepath = release_dir + "/" + std::to_string(allocation_id);
    std::ofstream release_file(release_filepath);
    release_file.close();

    return 0;
}
