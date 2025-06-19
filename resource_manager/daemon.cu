#include "resource_manager.h"
#include <mpi.h>
#include <cstdio>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_size < 2) {
        if (world_rank == 0) {
            fprintf(stderr, "Error: At least 2 processes are required (1 master + 1 worker).\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (world_rank == 0) {
        master_main_loop(world_size);
    } else {
        worker_main_loop(world_rank);
    }

    MPI_Finalize();
    return 0;
}
