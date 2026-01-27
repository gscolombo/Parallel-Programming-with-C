/*
    Program: Tree-Structure Global Sum (Assignment 3.3)

    This program implements a tree-structured algorithm to 
    compute a global sum across all processes. For simplicity, 
    each process contributes its own rank as the value to be 
    summed. Consequently, the final result can be verified by 
    comparing it to the closed-form sum of the first *p* 
    non-negative integers, where *p* is the total number of processes.
*/

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
        k = !i ? 1 : 2 << (i - 1); // Rank interval

        if (!(rank % k)) // Consider only processes with rank multiple of 2^i
        {
            /*
                Only processes with rank multiple of 2^(i + 1) and
                less than the number of nodes in the given height
                can receive operands
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
        printf("Sum of ranks = %g\nExpected = %i\n", total_sum, (n*(n-1))/2);
    }

    MPI_Finalize();

    return 0;
}