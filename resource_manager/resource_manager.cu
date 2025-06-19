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

    // The worker just waits for a shutdown signal from the master.
    // It doesn't interact with clients.
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

void handle_client_connection(MPI_Comm client_comm) {
    MPI_Status status;
    MPI_Probe(0, MPI_ANY_TAG, client_comm, &status);

    // Handle a request for a NEW allocation from this specific client
    if (status.MPI_TAG == TAG_REQUEST_ALLOCATION) {
        int num_gpus_requested;
        MPI_Recv(&num_gpus_requested, 1, MPI_INT, 0, TAG_REQUEST_ALLOCATION, client_comm, MPI_STATUS_IGNORE);
        printf("[Master] Received request for %d GPUs from a client.\n", num_gpus_requested);

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

                MPI_Send(&allocation_id, 1, MPI_INT, 0, TAG_GRANT_ALLOCATION, client_comm);
                MPI_Send(node.hostname.c_str(), node.hostname.length() + 1, MPI_CHAR, 0, TAG_GRANT_ALLOCATION, client_comm);
                MPI_Send(gpus_to_allocate.data(), gpus_to_allocate.size(), MPI_INT, 0, TAG_GRANT_ALLOCATION, client_comm);
                printf("[Master] Granted Allocation %d to client on Node %s.\n", allocation_id, node.hostname.c_str());

                allocated = true;
                break;
            }
        }
        if (!allocated) {
            printf("[Master] Could not satisfy request for %d GPUs. Rejecting.\n", num_gpus_requested);
            int rejection_signal = -1;
            MPI_Send(&rejection_signal, 1, MPI_INT, 0, TAG_REJECT_ALLOCATION, client_comm);
        }
    }
    // Handle a request to RELEASE an allocation
    else if (status.MPI_TAG == TAG_RELEASE_ALLOCATION) {
        int allocation_id_to_release;
        MPI_Recv(&allocation_id_to_release, 1, MPI_INT, 0, TAG_RELEASE_ALLOCATION, client_comm, MPI_STATUS_IGNORE);

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
    }
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

    // 2. Open a port and publish the service name
    char port_name[MPI_MAX_PORT_NAME];
    MPI_Open_port(MPI_INFO_NULL, port_name);
    MPI_Publish_name("gpu-manager-service", MPI_INFO_NULL, port_name);
    printf("[Master] Resource Manager is active and listening for clients.\n");
    printf("------------------------------------------------------------\n");

    // 3. Main loop: accept client connections and handle them
    while(true) {
        MPI_Comm client_comm;
        MPI_Comm_accept(port_name, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &client_comm);

        // Handle the connected client
        handle_client_connection(client_comm);

        // Disconnect from the client after handling one request/release cycle
        MPI_Comm_disconnect(&client_comm);
    }

    // Cleanup (in a real daemon, you'd have a signal handler to do this)
    MPI_Unpublish_name("gpu-manager-service", MPI_INFO_NULL, port_name);
    MPI_Close_port(port_name);
}
