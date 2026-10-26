/* Self-test for the CAGE PMPI shim. Known communication, known expected trace.
 * Run: make test   (needs 4 ranks)
 * Expected per step: ring sends r -> (r+1)%p of 1 MiB, one allreduce of 8 MiB over world,
 * one bcast of 1 MiB from rank 0 over a 2-rank sub-communicator {0,1}.
 * tests/test_pmpi_selftest.py checks the parsed trace against these numbers.
 */
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int rank, size, step;
    const int n1 = 1 << 18;          /* 1 MiB of floats */
    const int n8 = 1 << 21;          /* 8 MiB of floats */
    float *a, *b;
    MPI_Comm sub;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    a = calloc(n8, sizeof(float));
    b = calloc(n8, sizeof(float));
    MPI_Comm_split(MPI_COMM_WORLD, rank < 2 ? 0 : 1, rank, &sub);
    for (step = 0; step < 3; step++) {
        MPI_Sendrecv(a, n1, MPI_FLOAT, (rank + 1) % size, 0, b, n1, MPI_FLOAT,
                     (rank - 1 + size) % size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Allreduce(a, b, n8, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        if (rank < 2) MPI_Bcast(a, n1, MPI_FLOAT, 0, sub);
        MPI_Pcontrol(1);
    }
    MPI_Comm_free(&sub);
    free(a); free(b);
    MPI_Finalize();
    return 0;
}
