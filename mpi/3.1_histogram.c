#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

float *get_local_data(int rank, int p, int n, int *local_n, float *data) {
  float *local_data;
  int *sendcounts = (int *)malloc(p * sizeof(int));
  int *displacements;

  if (!sendcounts) {
    printf("Error during memory allocation for local N array.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (!rank) {
    int k = n / p;
    int r = n % p;

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
  local_data = (float *)malloc(*local_n * sizeof(float));

  if (!local_data) {
    printf("Error during memory allocation for local N array.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

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

  return local_data;
}

float *get_bin_maxes(float max_data, float min_data, int bins) {
  float *bin_maxes = (float *)malloc(bins * sizeof(float));

  if (!bin_maxes) {
    printf("Error during memory allocation for bins upper bounds.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  const float bin_width = (max_data - min_data) / bins;

  for (int i = 0; i < bins; i++)
    bin_maxes[i] = min_data + bin_width * (i + 1);

  return bin_maxes;
}

float *get_input(int *n, int *bins, float *max_data, float *min_data) {
  float *data;
  *max_data = -INFINITY;
  *min_data = INFINITY;

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

  *min_data = INFINITY;

  printf("Enter values separated by space: \n");
  for (int i = 0; i < *n; i++) {
    scanf("%f", &data[i]);

    // Find minimum and maximum data values
    if (data[i] < *min_data)
      *min_data = data[i];

    if (data[i] > *max_data)
      *max_data = data[i];
  }

  return data;
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
  int *bin_count, *local_bin_count;

  float *local_data, *bin_maxes, *data;
  float max_data, min_data;

  local_data = bin_maxes = NULL;
  bin_count = NULL;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (!rank) {
    int *bin_count = (int *)calloc(bins, sizeof(int));

    if (!bin_count) {
      printf("Error during memory allocation for global histogram.\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    data = get_input(&n, &bins, &max_data, &min_data);
  }

  // TODO: Use MPI derived datatype
  MPI_Bcast(&bins, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&max_data, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&min_data, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

  bin_maxes = get_bin_maxes(max_data, min_data, bins);

  local_bin_count = (int *)calloc(bins, sizeof(int));

  if (!local_bin_count) {
    printf("Error during memory allocation for local histogram.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  local_data = get_local_data(rank, p, n, &local_n, data);

  count_data(local_n, bins, local_bin_count, local_data, bin_maxes);

  MPI_Reduce(local_bin_count, bin_count, bins, MPI_INT, MPI_SUM, 0,
             MPI_COMM_WORLD);

  if (!rank)
    output_histogram(bins, bin_count, bin_maxes, min_data);

  free(bin_maxes);

  MPI_Finalize();

  return 0;
}