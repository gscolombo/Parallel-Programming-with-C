#include <libgen.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void write_output(char *fname, double *y, int n) {
  char *filename = basename(fname);
  filename = strtok(filename, ".");

  strcat(filename, ".out");

  FILE *out = fopen(filename, "w");

  for (int i = 0; i < n; i++)
    fprintf(out, "%lf\n", y[i]);

  fclose(out);
}

double *dot_product(double *A, double *x, int m, int n) {
  double *y = (double *)calloc(m, sizeof(double));

  if (!y) {
    printf("Error during memory allocation. Aborting. \n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      y[i] += A[i * n + j] + x[j];

  return y;
}

void get_input(char *fname, int rank, int p, int *n, double **local_A,
               double **local_x, int *local_n) {
  double *A, *x, *blk_cols;
  int local_A_n;
  FILE *f;

  if (!rank) {
    if (!(f = fopen(fname, "r"))) {
      printf("Error opening file. Aborting. \n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    fscanf(f, "%i", n);

    if ((*n <= 0)) {
      printf("Invalid input. Aborting. \n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    A = (double *)malloc(*n * *n * sizeof(double));
    x = (double *)malloc(*n * sizeof(double));

    if (!A || !x) {
      printf("Error during memory allocation. Aborting. \n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double start = MPI_Wtime();

    for (int i = 0; i < *n; i++)
      for (int j = 0; j < *n; j++)
        fscanf(f, "%lf", &A[i * *n + j]);

    double read_end_time = MPI_Wtime();
    printf("Read time (%i entries): %lf seconds\n", *n * *n,
           read_end_time - start);

    int i = 0;
    while (fscanf(f, "%lf", &x[i]) != EOF) {
      if (i > *n) {
        printf("Invalid input. Aborting. \n");
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      i++;
    }

    *local_n = *n / p;
    local_A_n = *local_n * *n;

    blk_cols = (double *)malloc(p * local_A_n * sizeof(double));
    if (!blk_cols) {
      printf("Error during memory allocation. Aborting. \n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    start = MPI_Wtime();
    int k = 0;
    for (int r = 0; r < p; r++)
      for (int i = 0; i < *n; i++)
        for (int j = r * *local_n; j < *local_n * (r + 1); j++)
          blk_cols[k++] = A[i * *n + j];

    double cvt_end_time = MPI_Wtime();
    printf("Conversion time (block-column): %lf seconds\n",
           cvt_end_time - start);
  }
  MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(local_n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  local_A_n = *local_n * *n;

  *local_x = (double *)malloc(*local_n * sizeof(double));
  *local_A = (double *)malloc(local_A_n * sizeof(double));

  if (!(*local_x) || !(*local_A)) {
    printf("Error during memory allocation. Aborting. \n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Scatter(x, *local_n, MPI_DOUBLE, *local_x, *local_n, MPI_DOUBLE, 0,
              MPI_COMM_WORLD);
  MPI_Scatter(blk_cols, local_A_n, MPI_DOUBLE, *local_A, local_A_n, MPI_DOUBLE,
              0, MPI_COMM_WORLD);

  if (!rank) {
    free(A);
    free(x);
    free(blk_cols);
  }
}

int main(int argc, char *argv[]) {
  if (argc == 1) {
    printf("Invalid number of inputs. Pass the path for a input file.\n");
    exit(1);
  }
  int rank, p, n, local_n;
  double *local_A, *local_x, *local_y, *y;

  local_A = local_x = y = NULL;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  get_input(argv[1], rank, p, &n, &local_A, &local_x, &local_n);

  double start = MPI_Wtime();
  local_y = dot_product(local_A, local_x, n, local_n);

  y = (double *)malloc(n * sizeof(double));

  if (!y) {
    printf("Error during memory allocation. Aborting. \n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Reduce(local_y, y, n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  double prod_end_time = MPI_Wtime() - start;
  double calc_time;

  MPI_Reduce(&prod_end_time, &calc_time, 1, MPI_DOUBLE, MPI_MIN, 0,
             MPI_COMM_WORLD);

  if (!rank) {
    printf("Minimum calculation time: %lf seconds\n", calc_time);
    write_output(argv[1], y, n);
  }

  MPI_Finalize();

  return 0;
}