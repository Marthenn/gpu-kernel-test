#ifndef RESOURCE_MANAGER_H
#define RESOURCE_MANAGER_H

#include <vector>
#include <string>
#include <map>

// --- MPI Message Tags ---
#define TAG_WORKER_REGISTER 1
#define TAG_REQUEST_ALLOCATION 2
#define TAG_GRANT_ALLOCATION 3
#define TAG_REJECT_ALLOCATION 4
#define TAG_RELEASE_ALLOCATION 5
#define TAG_SHUTDOWN 99

struct GPU {
    int id;
    std::string name;
    bool is_allocated;
    int allocation_id;
};

struct Node {
    int rank;
    std::string hostname;
    std::vector<GPU> gpus;
};

void master_main_loop(int world_size);
void worker_main_loop(int rank);

#endif // RESOURCE_MANAGER_H
