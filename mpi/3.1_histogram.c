#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void get_and_distr_input(int rank, int p, int *n, int *bins, int *local_n,
                         float *local_data, float *bin_maxes, float *min_data) {
  float *data;

  /* Read input data */
  if (!rank) {
    printf("N = ");
    fflush(stdout);
    scanf("%i", n);

    if (*n <= 0) {
      printf("Invalid input. N must be greater than or equal to 1.\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    printf("Number of bins: ");
    fflush(stdout);
    scanf("%i", bins);
    if (*bins <= 0) {
      printf("Invalid input. An histogram must have at least one bin.\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    data = (float *)malloc(*n * sizeof(float));
    if (!data) {
      printf("Error allocating memory for input data.");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    float max_data = -INFINITY;
    *min_data = INFINITY;

    printf("Enter values separated by space: \n");
    for (int i = 0; i < *n; i++) {
      scanf("%f", &data[i]);

      // Find minimum and maximum data values
      if (data[i] < *min_data)
        *min_data = data[i];

      if (data[i] > max_data)
        max_data = data[i];
    }

    printf("Minimum value: %g\n", *min_data);
    printf("Maximum value: %g\n", max_data);

    // Define upper bound for every bin
    bin_maxes = (float *)malloc(*bins * sizeof(float));

    if (!bin_maxes) {
      printf("Error during memory allocation for bins upper bounds.\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const float bin_width = (max_data - *min_data) / *bins;

    for (int i = 0; i < *bins; i++)
      bin_maxes[i] = *min_data + bin_width * (i + 1);
  }

  // Send number of bins and bins upper bounds to every process
  MPI_Bcast(bins, 1, MPI_COUNT, 0, MPI_COMM_WORLD);
  MPI_Bcast(bin_maxes, *bins, MPI_INT, 0, MPI_COMM_WORLD);

  /* Distribute data over process */
  int k, r;
  int *displacements;
  int *sendcounts = (int *)malloc(p * sizeof(int));

  if (!sendcounts) {
    printf("Error during memory allocation for local N array.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (!rank) {
    k = *n / p; // k data points for every process
    r = *n % p; // r < p remaining data points

    displacements = (int *)malloc(p * sizeof(int));

    if (!displacements) {
      printf(
          "Error during memory allocation for displacements in data array.\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Handle any remaining data to processes in rank order
    int c;
    for (int i = 0; i < p; i++, r--) {
      c = k + (r > 0);
      sendcounts[i] = c;
      displacements[i] = sendcounts[i] - c;
    }
  }

  // Send local N array for every process
  MPI_Bcast(sendcounts, p, MPI_INT, 0, MPI_COMM_WORLD);
  *local_n = sendcounts[rank];

  // Send data slice for each process
  if (!rank) {
    MPI_Scatterv(data, sendcounts, displacements, MPI_FLOAT, local_data,
                 *local_n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    free(data);
    free(displacements);
  } else
    MPI_Scatterv(data, sendcounts, displacements, MPI_FLOAT, local_data,
                 *local_n, MPI_FLOAT, 0, MPI_COMM_WORLD);

  free(sendcounts);
}

void count_data(int n, int bins, int *bin_count, float *data,
                float *bin_maxes) {
  for (int i = 0; i < n; i++) {
    for (int b = 0; b < bins; b++)
      if ((!b && data[i] <= bin_maxes[b]) ||
          ((bin_maxes[b - 1] < data[i]) && (data[i] <= bin_maxes[b])))
        bin_count[b]++;
  }
}

void output_histogram(int bins, int *bin_count, float *bin_maxes,
                      float min_data) {
  const char *fmt = "%g | %g : %i\n";
  printf("\n\n");
  printf(fmt, min_data, bin_maxes[0], bin_count[0]);
  for (int b = 1; b < bins; b++)
    printf(fmt, bin_maxes[b - 1], bin_maxes[1], bin_count[b]);
}

int main() {
  int n, bins, rank, p, local_n;
  float *local_data, *bin_maxes, min_data;
  int *bin_count, *local_bin_count;

  local_data = bin_maxes = NULL;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  get_and_distr_input(rank, p, &n, &bins, &local_n, local_data, bin_maxes,
                      &min_data);

  // Initialize bin count arrays
  if (!rank) {
    bin_count = (int *)calloc(bins, sizeof(int));
    if (!bin_count) {
      printf("Error during memory allocation for global histogram.\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  local_bin_count = (int *)calloc(bins, sizeof(int));

  if (!local_bin_count) {
    printf("Error during memory allocation for local histogram.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  count_data(local_n, bins, local_bin_count, local_data, bin_maxes);

  MPI_Reduce(local_bin_count, bin_count, bins, MPI_INT, MPI_SUM, 0,
             MPI_COMM_WORLD);

  if (!rank)
    output_histogram(bins, bin_count, bin_maxes, min_data);

  free(bin_maxes);

  MPI_Finalize();

  return 0;
}