// Matrix-Vector Multiplication with MPI and OpenMP
// Uses Row-wise matrix decomposition
//
// Author: Jimmy Nguyen
// Email: jimmy@jimmyworks.net
//
// Description:
// This C program performs Matrix-Vector Multiplication of a M x N matrix
// using up to 32 nodes using MPI and OpenMP parallelization for performance
// optimization.
// Performance tested and analyzed using Karp-Flatt metric in formal report.

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

// Main
// (input) int argc     main argument count
// (input) char** argv  main argument array
//                        argv[0] = program
//                        argv[1] = number of rows in matrix and vector length
//                        argv[2] = number of columns in matrix
//                        argv[3] = "DEBUG" (to print matrix-vector multiply results)
// (output) int         return code
// return code 0        success
// return code 1        failed usage due to incorrect args
// return code 2        failed due to too many processors for problem size
int main(int argc, char** argv) {

   int cols = 32768; // Number of columns
   int rows = 16384; // Number of rows and length of vector
   struct timeval start, stop; // start and stop timer
   int p_count; // Number of processes
   int p_id; // Get process rank

   int i, j; // general use iterators
   int *vector; // vector for MPI bcast
   int *matrix; // main matrix
   int *my_matrix; // partial matrix per processor
   int *result_vector; // result vector after MPI gatherv
   int debug = 0; // debug flag

   // Initialize and get values for MPI_COMM_WORLD
   MPI_Init(&argc, &argv); // Initialize MPI Environment
   MPI_Comm_size(MPI_COMM_WORLD, &p_count);  // Get the number of processes
   MPI_Comm_rank(MPI_COMM_WORLD, &p_id);  // Get the rank of the process

   // Check program args
   if(argc == 1)
   {
      if(!p_id) printf("Normal Run. Rows = %d, Cols = %d\n", rows, cols);
   }
   else if(argc == 3)
   {
      rows = atoi(argv[1]);
      cols = atoi(argv[2]);
      if(!p_id) printf("User-defined Run. Rows = %d, Cols = %d\n\n", rows, cols);
   }
   else if(argc == 4 && !strcmp(argv[3],"DEBUG"))
   {
      rows = atoi(argv[1]);
      cols = atoi(argv[2]);
      debug = 1;
      if(!p_id) printf("User-defined DEBUG Run. Rows = %d, Cols = %d\n\n", rows, cols);

   }
   else
   {
      if(!p_id) printf("Must run with no args or two args defining matrix dimensions.");
      MPI_Finalize();
      return 1;
   }

   // Allocate space for vector
   vector = malloc(rows*sizeof(int));

   //Create Matrix and Vector
   // Typically, this matrix would not be created here but already exist somewhere
   // in disk storage.  A manager process would have to divide the workload and
   // read pages from disk and distribute the data to workers accordingly.  Here,
   // we will simply create a matrix small enough to fit in memory to simulate this
   // algorithm.  Bear in mind that this generated matrix will already exist on all
   // processes, but we will still pass it to each process to simulate what would
   // actually happen in a large enough problem.
   if(!p_id)
   {
      // Allocate space for matrix and result vector on proc 0
      matrix = malloc(rows*cols*sizeof(int));
      result_vector = malloc(rows*sizeof(int));
      // Fill Vector
      for(i = 0; i < rows; i++)
         vector[i] = i+1;
      // Fill Matrix
      for(i=0; i < rows; i++)
      {
         for(j=0; j < cols; j++)
         {
            matrix[i*cols+j] = i+j;
         }
      }

   }

   // Start Timer
   MPI_Barrier(MPI_COMM_WORLD);
   gettimeofday( &start, NULL );   // Record start time

   // If more processors than rows, kill the program
   if(p_count > rows)
   {
      printf("Using too many processors for small problem size.  Exiting...\n");
      free(matrix);
      free(vector);
      free(result_vector);
      return 2;
   }
   // Organize work per block strip
   int block_rows[p_count];
   int block_cols[p_count];
   int row_remain = rows%p_count;
   for(i=0; i<p_count; i++)
   {
      block_rows[i] = rows/p_count;
      if(i < row_remain) block_rows[i]++;
      block_cols[i] = cols;
   }
   int my_rows = block_rows[p_id];
   int my_cols = block_cols[p_id];

   // Allocate all arrays for scatter and gather MPI routines
   my_matrix = malloc(my_rows*my_cols*sizeof(int)); // Partial matrices
   int *scount = malloc(p_count*sizeof(int)); // send counts for scatter
   int *displ = malloc(p_count*sizeof(int)); // displacement array for scatter
   int *r_displ = malloc(p_count*sizeof(int)); // displacement array for gather
   int *rcount = malloc(p_count*sizeof(int)); // return counts for gather
   int tail = 0; // place keeper for scatter displacement
   int r_tail = 0; // place keeper for gather displacement
   for(i=0; i<p_count; i++)
   {
      displ[i] = tail;
      r_displ[i] = r_tail;
      scount[i] = block_cols[i]*block_rows[i];
      rcount[i] = block_rows[i];
      r_tail = r_tail + rcount[i];
      tail = tail + scount[i];
   }

   // Scatter matrix
   MPI_Scatterv(matrix, scount, displ, MPI_INT, my_matrix, (my_rows*my_cols), MPI_INT, 0, MPI_COMM_WORLD);
   // Broadcast the vector
   MPI_Bcast(vector, rows, MPI_INT, 0, MPI_COMM_WORLD);

   // Do work
   int *partial_vector = calloc(block_rows[p_id], sizeof(int));
   int sum;

   // Optimize matrix multiplication with OpenMP
   #pragma omp parallel for reduction(+:sum)
   for(i=0; i<block_rows[p_id];i++)
   {
      sum = 0;
      for(j=0; j<block_cols[p_id]; j++)
      {
         // Sum reduction
         sum += my_matrix[i*block_cols[p_id]+j]*vector[j];
      }
      partial_vector[i] = sum;
   }

   // Gather the results
   MPI_Gatherv(partial_vector, block_rows[p_id], MPI_INT, result_vector, rcount, r_displ, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday( &stop, NULL );   // Record stop time
    float elapsed = ( (stop.tv_sec-start.tv_sec) +
                        (stop.tv_usec-start.tv_usec)/(float)1000000 );
    if(!p_id)
    {
        printf("Final time: %f\n", elapsed);

        // If in debug mode, print the matrix-vector multiplication
        if(debug)
        {
           // Print Result
           printf("Matrix\t\tVector\t\tResult\n");
           for(i=0; i < rows; i++)
           {
              printf("[");
              for(j=0; j < cols; j++)
              {
                 printf("%2d ", matrix[(i*cols)+j]);
              }
              printf("]\t[%2d]\t\t[%2d]\n", vector[i], result_vector[i]);
           }
        }
        // Free root-specific memory allocations
        free(matrix);
        free(vector);
        free(result_vector);
    }
    // Free up all allocations on all processes
    free(my_matrix);
    free(scount);
    free(rcount);
    free(r_displ);
    free(displ);

    MPI_Finalize();
}
