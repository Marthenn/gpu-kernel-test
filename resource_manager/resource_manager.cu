#include "resource_manager.h"
#include <iostream>
#include <unistd.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <wordexp.h>
#include <dirent.h>     // For reading directory contents
#include <sys/stat.h>   // For checking file existence
#include <algorithm>    // For std::remove

// ===================================================================
// ==                     WORKER DAEMON LOGIC                       ==
// ===================================================================
void worker_main_loop(int rank) {
    // Worker logic remains the same: register hardware and stay alive.
    char hostname[256];
    gethostname(hostname, 256);
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    MPI_Send(hostname, 256, MPI_CHAR, 0, TAG_WORKER_REGISTER, MPI_COMM_WORLD);
    MPI_Send(&device_count, 1, MPI_INT, 0, TAG_WORKER_REGISTER, MPI_COMM_WORLD);
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        MPI_Send(props.name, 256, MPI_CHAR, 0, TAG_WORKER_REGISTER, MPI_COMM_WORLD);
    }

    while (true) {
        MPI_Status status;
        MPI_Probe(0, TAG_SHUTDOWN, MPI_COMM_WORLD, &status);
        break;
    }
}


// ===================================================================
// ==                  MASTER DAEMON LOGIC (SERVER)                 ==
// ===================================================================
std::vector<Node> cluster_nodes;
int next_allocation_id = 100;

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

// Create directories if they don't exist
void ensure_dir_exists(const std::string& path) {
    mkdir(path.c_str(), 0777);
}

void master_main_loop(int world_size) {
    // 1. Register all workers
    for (int rank = 1; rank < world_size; ++rank) {
        Node n;
        n.rank = rank;
        char hostname[256];
        int device_count;
        MPI_Recv(hostname, 256, MPI_CHAR, rank, TAG_WORKER_REGISTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        n.hostname = std::string(hostname);
        MPI_Recv(&device_count, 1, MPI_INT, rank, TAG_WORKER_REGISTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < device_count; ++i) {
            char gpu_name[256];
            MPI_Recv(gpu_name, 256, MPI_CHAR, rank, TAG_WORKER_REGISTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            n.gpus.push_back({i, std::string(gpu_name), false, -1});
        }
        cluster_nodes.push_back(n);
    }

    // 2. Setup communication directories
    std::string req_dir = expand_path("~/gpu_requests");
    std::string resp_dir = expand_path("~/gpu_responses");
    std::string release_dir = expand_path("~/gpu_releases");
    ensure_dir_exists(req_dir);
    ensure_dir_exists(resp_dir);
    ensure_dir_exists(release_dir);

    printf("[Master] Resource Manager is active.\n");
    printf("[Master] Watching for requests in: %s\n", req_dir.c_str());
    printf("------------------------------------------------------------\n");

    // 3. Main polling loop
    while(true) {
        // --- Check for new allocation requests ---
        DIR* dir = opendir(req_dir.c_str());
        if (dir) {
            struct dirent* entry;
            while ((entry = readdir(dir)) != NULL) {
                std::string filename = entry->d_name;
                if (filename[0] == '.') continue; // Skip . and ..

                std::string req_filepath = req_dir + "/" + filename;
                std::ifstream infile(req_filepath);
                if (infile.is_open()) {
                    int num_gpus_requested;
                    infile >> num_gpus_requested;
                    infile.close();
                    printf("[Master] Processing request '%s' for %d GPUs.\n", filename.c_str(), num_gpus_requested);

                    bool allocated = false;
                    for (auto& node : cluster_nodes) {
                        std::vector<int> free_gpu_ids;
                        for (const auto& gpu : node.gpus) {
                            if (!gpu.is_allocated) free_gpu_ids.push_back(gpu.id);
                        }

                        if (free_gpu_ids.size() >= num_gpus_requested) {
                            int allocation_id = next_allocation_id++;
                            std::vector<int> gpus_to_allocate(free_gpu_ids.begin(), free_gpu_ids.begin() + num_gpus_requested);
                            for (int gid : gpus_to_allocate) {
                                node.gpus[gid].is_allocated = true;
                                node.gpus[gid].allocation_id = allocation_id;
                            }

                            // Write response file
                            std::string resp_filepath = resp_dir + "/" + filename;
                            std::ofstream outfile(resp_filepath);
                            outfile << allocation_id << std::endl;
                            outfile << node.hostname << std::endl;
                            for (int gid : gpus_to_allocate) outfile << gid << " ";
                            outfile << std::endl;
                            outfile.close();

                            printf("[Master] Granted Allocation %d to client on Node %s.\n", allocation_id, node.hostname.c_str());
                            allocated = true;
                            break;
                        }
                    }
                    if (!allocated) {
                        printf("[Master] Could not satisfy request. Rejecting.\n");
                        std::string resp_filepath = resp_dir + "/" + filename;
                        std::ofstream outfile(resp_filepath);
                        outfile << "REJECTED" << std::endl;
                        outfile.close();
                    }
                    std::remove(req_filepath.c_str()); // Delete processed request
                }
            }
            closedir(dir);
        }

        // --- Check for release requests ---
        dir = opendir(release_dir.c_str());
        if (dir) {
            struct dirent* entry;
            while ((entry = readdir(dir)) != NULL) {
                std::string filename = entry->d_name;
                if (filename[0] == '.') continue;

                int allocation_id_to_release = std::stoi(filename);
                 bool found = false;
                for (auto& node : cluster_nodes) {
                    for (auto& gpu : node.gpus) {
                        if (gpu.is_allocated && gpu.allocation_id == allocation_id_to_release) {
                            gpu.is_allocated = false;
                            gpu.allocation_id = -1;
                            found = true;
                        }
                    }
                }
                if (found) {
                    printf("[Master] Released Allocation %d.\n", allocation_id_to_release);
                }
                std::string release_filepath = release_dir + "/" + filename;
                std::remove(release_filepath.c_str());
            }
            closedir(dir);
        }

        usleep(500000); // Poll every 0.5 seconds
    }
}
