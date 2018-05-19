// Matrix-Vector Multiplication
// CHECKERBOARD BLOCK-DECOMPOSITION 
// WORK-IN-PROGRESS (PARTIAL SOLUTION)
//
// NOTE: see matrix_vector_rowwise.c for complete solution using
//       row-wise decomposition
//
// Author: Jimmy Nguyen
// Email: jimmy@jimmyworks.net
//
// Description:
// This C program performs Matrix-Vector Multiplication of a M x N matrix
// using up to 32 nodes using MPI and OpenMP parallelization for performance
// optimization.

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


int main(int argc, char** argv) {

   int m = 32768; // Number of columns
   int n = 16384; // Number of rows and length of vector
   double timer; // Execution time
   int p_count; // Number of processes
   int p_id; // Get process rank

   int i, j;
   MPI_Comm Cart_Comm;
   MPI_Status status;
   int periodic[2];
   int dim_size[2];
   int coord[2];
   int grid_id;
   int rows, cols;
   int first_row;
   int last_row;
   int first_col;
   int last_col;
   int *vector;
   int *matrix;
   int *my_matrix;

   MPI_Init(&argc, &argv); // Initialize MPI Environment
   MPI_Comm_size(MPI_COMM_WORLD, &p_count);  // Get the number of processes
   MPI_Comm_rank(MPI_COMM_WORLD, &p_id);  // Get the rank of the process


   if(argc == 1)
   {
      if(!p_id) printf("Normal Run. M = %d, N = %d\n", m, n);
   }
   else if(argc == 3)
   {
      m = atoi(argv[1]);
      n = atoi(argv[2]);
      if(!p_id) printf("User-defined Run. M = %d, N = %d\n", m, n);
   }
   else
   {
      if(!p_id) printf("Must run with no args or two args defining matrix dimensions.");
      MPI_Finalize();
      return 1;
   }

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
      matrix = malloc(m*n*sizeof(int));
      vector = malloc(m*sizeof(int));

      // Fill Vector
      for(i = 0; i < m; i++)
         vector[i] = i+1;
      // Fill Matrix
      for(i=0; i < m; i++)
      {
         for(j=0; j < n; j++)
         {
            matrix[(i*m-1)+j] = i+j;
         }
      }

      // Print Two items
      for(i=0; i < m; i++)
      {
         printf("[");
         for(j=0; j < n; j++)
         {
            printf("%d ", matrix[(i*m-1)+j]);
         }
         printf("]\t[%d]\n", vector[i]);
      }

   }

   // Start Timer
   timer = -MPI_Wtime();

   // Create Cartesian Communicator

    int dim_rows = (int)floor(sqrt((float)p_count));

    while(p_count%dim_rows){
       if(!p_id) printf("%d\n", dim_rows);
       dim_rows--;
    }
    int dim_cols = p_count/dim_rows;
    if(!p_id) printf("Block rows: %d Block cols: %d\n", dim_rows, dim_cols);

    dim_size[0] = dim_rows;
    dim_size[1] = dim_cols;
    periodic[0] = periodic[1] = 0;
    MPI_Dims_create(p_count, 2, dim_size);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim_size, periodic, 1, &Cart_Comm);
    MPI_Comm_rank(Cart_Comm, &grid_id);
    MPI_Cart_coords(Cart_Comm, grid_id, 2, coord);


   // Calcuate work to be divided
   rows = m/dim_rows;
   cols = n/dim_rows;
   int row_remain = m%dim_rows;
   int col_remain = n%dim_cols;

/*   // Making block types
   MPI_Datatype blocktype;
   MPI_Datatype blocktype2;

   MPI_Type_vector(block_rows, block_cols, n, MPI_INT, &blocktype2);
   MPI_Type_create_resized( blocktype2, 0, sizeof(int), &blocktype);
   MPI_Type_commit(&blocktype);
*/
      // Organize block_rows per block
      int block_rows[dim_rows];
      for(i=0; i<dim_rows; i++)
      {
         block_rows[i] = rows;
         if(i < row_remain) block_rows[i]++;
      }

      // Organize block_n per block
      int block_cols[dim_cols];
      for(i=0; i<dim_rows; i++)
      {
         block_cols[i] = cols;
         if(i < col_remain) block_cols[i]++;
      }

   // All processes create a matrix 1 size larger with 1 extra column for the vector
   int my_rows = block_rows[coord[0]];
   int my_cols = block_cols[coord[1]];

   my_matrix = malloc(my_rows*my_cols*sizeof(int));
   int *my_vector = malloc(my_rows*sizeof(int));

   // Print the process row/col-wise decomposition
   if(!p_id)
   {
      printf("Row-wise decomp:\n");
      for(i=0; i<dim_rows; i++)
      {
        printf("%d ", block_rows[i]);
      }
      printf("\nColumn-wise decomp:\n");
      for(i=0; i<dim_cols; i++)
      {
        printf("%d ", block_cols[i]);
      }
      printf("\n");
   }

   MPI_Comm Root_Comm;
   MPI_Comm Col_Comm;

   int root_color, col_color;
   if(!coord[0]) root_color = 1;
   else root_color = MPI_UNDEFINED;
   col_color = coord[1];

   MPI_Comm_split(Cart_Comm, root_color, grid_id, &Root_Comm);
   MPI_Comm_split(Cart_Comm, col_color, grid_id, &Col_Comm);

   int root_comm_id;
   int col_comm_id;
   MPI_Comm_rank(Col_Comm, &col_comm_id);
   if(!coord[0])
   {
      MPI_Comm_rank(Root_Comm, &root_comm_id);
      printf("World id: %d, Cartesian id: %d, Root Comm id: %d, Col Comm id:%d, Coords: %d,%d\n",
           p_id, grid_id, root_comm_id, col_comm_id, coord[0], coord[1]);
   }
   else
   {
      printf("World id: %d, Cartesian id: %d, Col Comm id:%d, Coords: %d,%d\n",
           p_id, grid_id, col_comm_id, coord[0], coord[1]);
   }

   // Work needs to be divided evenly
   // May need to defined a new MPI_Type
   // TODO: Complete Checkerboard decomposition implementation

    free(my_matrix);
    timer += MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    if(!p_id)
    {
        free(matrix);
        free(vector);
        printf("Dim1: %d, Dim2: %d\n", dim_rows, dim_cols);
        //printf("Final time: %d\n", timer);
    }
    MPI_Finalize();
}
