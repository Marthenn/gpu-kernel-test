#include "resource_manager.h"
#include <iostream>
#include <unistd.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>

// ===================================================================
// ==                     WORKER DAEMON LOGIC                       ==
// ===================================================================
void worker_main_loop(int rank) {
    // Discover local hardware
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    // Register to the master
    MPI_Send(hostname, 256, MPI_CHAR, 0, TAG_WORKER_REGISTER, MPI_COMM_WORLD);
    MPI_Send(&device_count, 1, MPI_INT, 0, TAG_WORKER_REGISTER, MPI_COMM_WORLD);
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp props{};
        cudaGetDeviceProperties(&props, i);
        MPI_Send(props.name, 256, MPI_CHAR, 0, TAG_WORKER_REGISTER, MPI_COMM_WORLD);
    }
    printf("[Worker %d]  Registered with master.\n", rank);

    // Stay alive for heartbeat
    while (true) {
        MPI_Status status;
        MPI_Probe(0, TAG_SHUTDOWN, MPI_COMM_WORLD, &status);
        printf("[Worker %d] Received shutdown. Exiting.\n", rank);
        break;
    }
}


// ===================================================================
// ==                  MASTER DAEMON LOGIC (BROKER)                 ==
// ===================================================================
std::vector<Node> cluster_nodes;
int next_allocation_id = 100;

void master_main_loop(int world_size) {
    // Register all workers
    for (int rank = 1; rank < world_size; rank++) {
        Node n;
        n.rank = rank;
        char hostname[256];
        int device_count;
        MPI_Recv(hostname, 256, MPI_CHAR, rank, TAG_WORKER_REGISTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        n.hostname = std::string(hostname);
        MPI_Recv(&device_count, 1, MPI_INT, rank, TAG_WORKER_REGISTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0 ; i < device_count; i++) {
            char gpu_name[256];
            MPI_Recv(gpu_name, 256, MPI_CHAR, rank, TAG_WORKER_REGISTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            n.gpus.push_back({i, std::string(gpu_name), false, -1});
        }
        cluster_nodes.push_back(n);
        printf("[Master] Registered worker %d: %s with %d GPUs.\n", rank, n.hostname.c_str(), device_count);
    }
    printf("[Master] All workers registered. Total nodes: %zu\n", cluster_nodes.size());
    printf("[Master] Entering main loop...\n");

    // Listen for allocation/release requests from clients
    while (true) {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        // NEW ALLOCATION
        if (status.MPI_TAG == TAG_REQUEST_ALLOCATION) {
            int num_gpus_requested;
            MPI_Recv(&num_gpus_requested, 1, MPI_INT, status.MPI_SOURCE, TAG_REQUEST_ALLOCATION, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("[Master] Received allocation request for %d GPUs from client %d.\n", num_gpus_requested, status.MPI_SOURCE);

            bool allocated = false;
            for (auto& node : cluster_nodes) {
                std::vector<int> free_gpu_ids;
                for (const auto& gpu: node.gpus) {
                    if (!gpu.is_allocated) {
                        free_gpu_ids.push_back(gpu.id);
                    }
                }

                if (free_gpu_ids.size() >= num_gpus_requested) {
                    const int allocation_id = next_allocation_id++;
                    std::vector<int> gpus_to_allocate(free_gpu_ids.begin(), free_gpu_ids.begin() + num_gpus_requested);

                    for (int gid : gpus_to_allocate) {
                        node.gpus[gid].is_allocated = true;
                        node.gpus[gid].allocation_id = allocation_id;
                    }

                    // GRANT ALLOCATION
                    MPI_Send(&allocation_id, 1, MPI_INT, status.MPI_SOURCE, TAG_GRANT_ALLOCATION, MPI_COMM_WORLD);
                    MPI_Send(node.hostname.c_str(), node.hostname.length()+1, MPI_CHAR, status.MPI_SOURCE, TAG_GRANT_ALLOCATION, MPI_COMM_WORLD);
                    MPI_Send(gpus_to_allocate.data(), gpus_to_allocate.size(), MPI_INT, status.MPI_SOURCE, TAG_GRANT_ALLOCATION, MPI_COMM_WORLD);
                    printf("[Master] Granted allocation %d to client %d on node %s with GPUs: ", allocation_id, status.MPI_SOURCE, node.hostname.c_str());

                    allocated = true;
                    break;
                }
            }

            if (!allocated) {
                printf("[Master] No available resources for client %d. Sending failure.\n", status.MPI_SOURCE);
                int rejection_signal = -1;
                MPI_Send(&rejection_signal, 1, MPI_INT, status.MPI_SOURCE, TAG_REJECT_ALLOCATION, MPI_COMM_WORLD);
            }
        }

        // HANDLE RELEASE
        else if (status.MPI_TAG == TAG_RELEASE_ALLOCATION) {
            int allocation_id_to_release;
            MPI_Recv(&allocation_id_to_release, 1, MPI_INT, status.MPI_SOURCE, TAG_RELEASE_ALLOCATION, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

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
                printf("[Master] Released allocation %d from client %d.\n", allocation_id_to_release, status.MPI_SOURCE);
            }
        }
    }
}
