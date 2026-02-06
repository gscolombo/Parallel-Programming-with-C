#include <mpi.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ULONG unsigned long long

ULONG calc_pi_estimate(ULONG num_tosses, int rank) {
  double x, y;
  ULONG num_in_circle = 0;

  std::random_device rd;
  std::mt19937 rng(rd() ^ (rank << 16));

  std::uniform_real_distribution<double> dist(-1, 1);

  for (ULONG i = 0; i < num_tosses; i++) {
    x = dist(rng); // -1 <= x <= 1
    y = dist(rng); // -1 <= y <= 1

    if ((x * x + y * y) <= 1)
      num_in_circle++;
  }

  return num_in_circle;
}

int main() {
  ULONG tosses, local_tosses, n, local_n;
  int rank, p;

  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  if (!rank) {
    printf("Enter number of tosses: ");
    fflush(stdout);

    if (!scanf("%llu", &tosses)) {
      printf("Error reading input. Aborting.\n");
      MPI_Abort(MPI_COMM_WORLD, 0);
    }
  }

  MPI_Bcast(&tosses, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  ULONG r = tosses % p;
  if (!r)
    local_tosses = tosses / p;
  else
    local_tosses = tosses / p + ((ULONG)rank < r);

  if (!rank)
    printf("Estimating π. This might take some time...\n");

  local_n = calc_pi_estimate(local_tosses, rank);
  MPI_Reduce(&local_n, &n, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0,
             MPI_COMM_WORLD);

  if (!rank)
    printf("π ≈ %g\n", 4 * n / (double)tosses);

  MPI_Finalize();

  return 0;
}