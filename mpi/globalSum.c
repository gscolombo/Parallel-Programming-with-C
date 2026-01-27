#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main()
{
    double h, total_sum, local_sum;
    int n, rank, k;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n);

    h = ceil(log2((double)n)); // Tree height

    if (!rank)
    {
        printf("Number of process: %i\n", n);
        printf("Tree height: %i\n", (int)h);
    }

    total_sum = rank;
    local_sum = 0;

    for (int i = 0; i < h; i++)
    {
        k = !i ? 1 : 2 << (i - 1); // Rank offset

        if (!(rank % k)) // Consider only process with ranks multiple of 2^i
        {
            /*
                Only processes with rank multiple of 2^(i + 1) and
                less than the current rank offset must receive operands
            */
            if (!(rank % (2 << i)) && rank < (n - k))
            {
                MPI_Recv(&local_sum, 1, MPI_DOUBLE, rank + k, 0, MPI_COMM_WORLD, NULL);
                total_sum += local_sum;

                continue;
            }

            MPI_Send(&total_sum, 1, MPI_DOUBLE, rank - k, 0, MPI_COMM_WORLD);
        }
    }

    if (!rank)
    {
        printf("Sum of ranks = %g\n", total_sum);
    }

    MPI_Finalize();

    return 0;
}