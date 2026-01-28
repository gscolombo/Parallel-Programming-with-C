#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int *hist(int n, float data[], int bins, float max_meas, float min_meas) {
  // Define upper bound for every bin
  const float bin_width = (max_meas - min_meas) / bins;

  float *bins_maxes = (float *)calloc(bins, sizeof(float));
  int *bin_count = (int *)calloc(bins, sizeof(int));

  if (!bins_maxes || !bin_count) {
    printf("Error during memory allocation for histogram data.\n");
    return NULL;
  }

  for (int i = 0; i < bins; i++) {
    bins_maxes[i] = min_meas + bin_width * (i + 1);
    bin_count[i] = 0;
  }
    
  // Distribute data into bins
  int rank, p;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  

  MPI_Finalize();

  free(bins_maxes);

  return bin_count;
}

int main() {
  int n, bins;
  float max_meas, min_meas;
  float *data;

  scanf("N = %i\n", &n);
  if (n <= 0) {
    printf("Invalid input. N must be greater than or equal to 1.\n");
    return 1;
  }

  scanf("Number of bins = %i", &bins);
  if (bins <= 0) {
    printf("Invalid input. An histogram must have at least one bin.\n");
    return 1;
  }

  data = (float *)calloc(n, sizeof(float));
  if (!data) {
    printf("Error allocating memory for input data.");
    return 1;
  }

  max_meas = -INFINITY;
  min_meas = INFINITY;
  for (int i = 0; i < n; i++) {
    scanf("%f ", &data[i]);

    if (data[i] < min_meas)
      min_meas = data[i];

    if (data[i] > max_meas)
      max_meas = data[i];
  }

  int *histogram = hist(n, data, bins, max_meas, min_meas);

  free(data);
  free(histogram);

  return 0;
}