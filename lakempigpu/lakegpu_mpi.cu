/******************************************************************************
* FILE: lakegpu_mpi.cu
*
* Group Info:
* agoel5 Anshuman Goel
* kgondha Kaustubh Gondhalekar
* ndas Neha Das
*
* LAST REVISED: 9/19/2017
******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <mpi.h>

#define VSQR 0.1
#define TSCALE 1.0

#define __DEBUG

#define CUDA_CALL( err )     __cudaSafeCall( err, __FILE__, __LINE__ )
#define CUDA_CHK_ERR() __cudaCheckError(__FILE__,__LINE__)

/**************************************
* void __cudaSafeCall(cudaError err, const char *file, const int line)
* void __cudaCheckError(const char *file, const int line)
*
* These routines were taken from the GPU Computing SDK
* (http://developer.nvidia.com/gpu-computing-sdk) include file "cutil.h")
**************************************/

extern int tpdt(double *t, double dt, double end_time);


//device function for the device ( this is same as function f in lake_mpi.cu )
__device__ double fn(double p, double t){

  return -expf(-TSCALE * t) * p;
}


//error checking function for cuda calls
inline void __cudaSafeCall( cudaError err, const char *file, const int line ){

  #ifdef __DEBUG
  #pragma warning( push )
  #pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
    do{
      if ( cudaSuccess != err ){

        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
      }
    } while ( 0 );
  #pragma warning( pop )
  #endif  // __DEBUG
  return;
}


//error checking function for cuda calls
inline void __cudaCheckError( const char *file, const int line ){

  #ifdef __DEBUG
  #pragma warning( push )
  #pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
    do{
      cudaError_t err = cudaGetLastError();
      if ( cudaSuccess != err ){

        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s.\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
      }
      // More careful checking. However, this will affect performance.
      // Comment if not needed.
      /*err = cudaThreadSynchronize();
      if( cudaSuccess != err )
      {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
      }*/
    } while ( 0 );
    #pragma warning( pop )
    #endif // __DEBUG
    return;
}



//updates the grid state from time t to time t+dt
__global__ static void evolve(double *un, double *uc, double *uo,
                              double *pebbles, int *n, double *h, double *dt,
                              double *t, int *n_blocks, int *n_threads, int *rank){

  int i, j;
  unsigned int bid = blockDim.x * blockIdx.x + threadIdx.x;

  //set the origin grid points for each quadrant for every processor node
  if(*rank == 0){
    i = (bid / (*n/2));
    j = (bid % (*n/2));
  }
  else if (*rank == 1){

    i = (bid / (*n/2));
    j = (bid % (*n/2)+ *n/2);
  }
  else if (*rank == 2){

    i = (bid / (*n/2)+ *n/2);
    j = (bid % (*n/2));
  }
  else if (*rank == 3){

    i = (bid / (*n/2)+ *n/2);
    j = (bid % (*n/2)+ *n/2);
  }

  //convert a 2D matrix co-ordinates into a 1-D matrix co-ordinate
  int idx = i * *n + j;

  //values at lake edge points are set to zero
  if( i == 0 || i == *n - 1 || j == 0 || j == *n - 1
      || i == *n - 2 || i == 1 || j == *n - 2 || j == 1){

    un[idx] = 0.;
  }
  else{
    //compute the 13-point stencil function for every grid point
    un[idx] = 2*uc[idx] - uo[idx] + VSQR *(*dt * *dt) *
    ((  uc[idx-1] // WEST
        + uc[idx+1] // EAST
        + uc[idx + *n] // SOUTH
        + uc[idx - *n] // NORTH
       + 0.25*( uc[idx - *n - 1 ] // NORTHWEST
              + uc[idx - *n + 1 ] // NORTHEAST
              + uc[idx + *n - 1 ] // SOUTHWEST
              + uc[idx + *n + 1 ] // SOUTHEAST
              )
      + 0.125*( uc[idx - 2 ]  // WESTWEST
              + uc[idx + 2 ] // EASTEAST
              + uc[idx - 2 * *n ] // NORTHNORTH
              + uc[idx + 2 * *n ] // SOUTHSOUTH
              )
      - 6 * uc[idx])/(*h * *h) + fn(pebbles[idx],*t));
    }

    __syncthreads();

    //save most recent two time-stamps into uo and uc
    uo[idx] = uc[idx];
    uc[idx] = un[idx];
    //move the timestamp forward by dt
    (*t) = (*t) + *dt;
}

//sends 2 rows of length k ( = n/2 when using it in MPI grid communication)
//starting at point (start_row, start_col)
void Send_Row(int start_row, int start_col, int k, int dest, double *u){

  double msg[2][ k];
  for(int i = start_row; i <= start_row + 1; i++){
    for(int j = 0; j <= k-1; j++){

      msg[i - start_row][j] = u[ i*2*k + j + start_col];
    }
  }
  MPI_Send(msg, 2 * k, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
}

//sends 2 columns of length k ( = n/2 when using it in MPI grid communication)
//starting at point (start_row, start_col)
void Send_Col(int start_row, int start_col, int k, int dest, double *u){

  double msg[k][2];
  for(int i = 0; i <= k-1; i++){
    for(int j = start_col; j <= start_col + 1; j++){
      msg[i][j - start_col] = u[ (i + start_row)*2*k + j ];
    }
  }
  MPI_Send(msg, 2 * k, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
}

//receives 2 rows of length k ( = n/2 when using it in MPI grid communication)
//starting at point (start_row, start_col)
void Recv_Row(int start_row, int start_col, int k, int source, double *u){

  double msg[2][k];
  MPI_Recv(msg, 2*k, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  for(int i = start_row; i <= start_row + 1; i++){
    for(int j = 0; j <= k-1; j++){
      u[i*2*k + j + start_col] = msg[i - start_row][j] ;
    }
  }
}


//receives 2 columns of length k ( = n/2 when using it in MPI grid communication)
//starting at point (start_row, start_col)
void Recv_Col(int start_row, int start_col, int k, int source, double *u)
{
  double msg[k][2];
  MPI_Recv(msg, 2*k, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  for(int i = 0; i <= k-1; i++){
    for(int j = start_col; j <= start_col + 1; j++){
       u[(i+start_row) * 2 * k + j ] = msg[i][j - start_col];
    }
  }
}


// simulates the state of the grid after the given time, using a 13-point stencil function
void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n,
            double h, double end_time, int nthreads, int rank){
	cudaEvent_t kstart, kstop;
	float ktime;

	/* HW2: Define your local variables here */

  double t=0.0;
  double dt = h / 2.0;
  //compute the number of blocks for every GPU for a MPI node
  int blocks = (int)pow(n / nthreads, 2) / 4;
  //compute the number of threads per block for every GPU for a MPI node
  int threads = nthreads * nthreads;

  //boundary co-ordinates for every quadrant
  //including the additional rows and columns required for edge points
  //to compute the 13 point stencil function
  int r_min, r_max, c_min, c_max;

  //device copies of the host variables
  int *blocks_d, *threads_d, *n_d, *rank_d;
  double *un_d, *uc_d, *uo_d, *pebs_d, *t_d, *dt_d, *h_d;

  if (nthreads > n){
    printf("Please select a thread number less than the grid dimension.\n");
    return;
  }

  //allocate memory for the device variables
  cudaMalloc( (void **) &un_d, sizeof(double) * n * n);
  cudaMalloc( (void **) &uc_d, sizeof(double) * n * n);
  cudaMalloc( (void **) &uo_d, sizeof(double) * n * n);
  cudaMalloc( (void **) &pebs_d, sizeof(double) * n * n);
  cudaMalloc( (void **) &blocks_d, sizeof(int) * 1 );
  cudaMalloc( (void **) &threads_d, sizeof(int) * 1 );
  cudaMalloc( (void **) &n_d, sizeof(int) * 1 );
  cudaMalloc( (void **) &t_d, sizeof(double) * 1 );
  cudaMalloc( (void **) &dt_d, sizeof(double) * 1 );
  cudaMalloc( (void **) &h_d, sizeof(double) * 1 );
  cudaMalloc( (void **) &rank_d, sizeof(int) * 1 );

  /* Set up device timers */
	CUDA_CALL(cudaSetDevice(0));
	CUDA_CALL(cudaEventCreate(&kstart));
	CUDA_CALL(cudaEventCreate(&kstop));

	/* Start GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstart, 0));

  //copy host variables to device
  CUDA_CALL(cudaMemcpy( uc_d, u1, sizeof(double) * n * n, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( un_d, u, sizeof(double) * n * n, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( uo_d, u0, sizeof(double) * n * n, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( pebs_d, pebbles, sizeof(double) * n * n, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( blocks_d, &blocks, sizeof(int) * 1, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( threads_d, &threads, sizeof(int) * 1, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( n_d, &n, sizeof(int) * 1, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( h_d, &h, sizeof(double) * 1, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( dt_d, &dt, sizeof(double) * 1, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( t_d, &t, sizeof(double) * 1, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( rank_d, &rank, sizeof(int) * 1, cudaMemcpyHostToDevice ));


  //compute boundary variables for each node processor
  if(rank == 0){
    r_min = 0;
    r_max = n / 2 + 1;
    c_min = 0;
    c_max = n / 2 + 1;
  }
  else if (rank == 1){
    r_min = 0;
    r_max = n / 2 + 1;
    c_min = n / 2 - 2;
    c_max = n - 1;
  }
  else if (rank == 2){
    r_min = n / 2 - 2;
    r_max = n - 1;
    c_min = 0;
    c_max = n / 2 + 1;
  }
  else if (rank == 3){
    r_min = n / 2 - 2;
    r_max = n - 1;
    c_min = n / 2 - 2;
    c_max = n - 1;
  }


  //compute state of the grid over the given time
  while(1){

    evolve<<< blocks, threads >>>(un_d, uc_d, uo_d, pebs_d, n_d, h_d, dt_d,
                                  t_d, blocks_d, threads_d, rank_d);

    CUDA_CALL(cudaMemcpy( u, un_d, sizeof(double) * n * n, cudaMemcpyDeviceToHost ));

    //exit from the loop if time exceeds final timestamp
    if(!tpdt(&t,dt,end_time)){
      break;
    }

    //MPI communication between nodes for
    //the exchange of the extra rows and columns and diagonal points
    if (rank == 0){
      //node 0 sends and receives rows and columns from 1 and 2
      Send_Col(0, c_max-3, n/2, 1, u);
      Send_Row(r_max-3, 0, n/2, 2, u);
      Recv_Col(0, c_max-1, n/2, 1, u);
      Recv_Row(r_max-1, 0, n/2, 2, u);
      //node 0 sends and receives diagonal points from 3
      MPI_Send(u+(r_max-2)*n+c_max-2, 1, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD);
      MPI_Recv(u+(r_max-1)*n+c_max-1, 1, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else if(rank == 1){
      //node 1 sends and receives rows and columns from 0 and 3
      Recv_Col(0, c_min, n/2, 0, u);
      Send_Col(0, c_min+2, n/2, 0, u);
      Send_Row(r_max-3, n/2, n/2, 3, u);
      Recv_Row(r_max-1, n/2, n/2, 3, u);
      //node 1 sends and receives diagonal points from 2
      MPI_Send(u+(r_max-2)*n+c_min+2, 1, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD);
      MPI_Recv(u+(r_max-1)*n+c_min+1, 1, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else if(rank == 2){
      //node 2 sends and receives rows and columns from 0 and 3
      Recv_Row(r_min, 0, n/2, 0, u);
      Send_Row(r_min+2, 0, n/2, 0, u);
      Send_Col(n/2, c_max-3, n/2, 3, u);
      Recv_Col(n/2, c_max-1, n/2, 3, u);
      //node 2 sends and receives diagonal points from 1
      MPI_Recv(u+(r_min+1)*n+c_max-1, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(u+(r_min+2)*n+c_max-2, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    }
    else if(rank == 3){
      //node 3 sends and receives rows and columns from 1 and 2
      Recv_Row(r_min, n/2, n/2, 1, u);
      Recv_Col(n/2, c_min, n/2, 2, u);
      Send_Row(r_min+2, n/2, n/2, 1, u);
      Send_Col(n/2, c_min+2, n/2, 2, u);
      //node 3 sends and receives diagonal points from 0
      MPI_Recv(u+(r_min+1)*n+c_min+1, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(u+(r_min+2)*n+c_min+2, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    CUDA_CALL(cudaMemcpy( uc_d, u, sizeof(double) * n * n, cudaMemcpyHostToDevice ));
  }


  /* Stop GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstop, 0));
	CUDA_CALL(cudaEventSynchronize(kstop));
	CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));

  if(rank==0){
	 printf("GPU computation: %f msec\n", ktime);
  }

  //free resources
  cudaFree(un_d);
  cudaFree(uc_d);
  cudaFree(uo_d);
  cudaFree(blocks_d);
  cudaFree(threads_d);
  cudaFree(pebs_d);
  cudaFree(n_d);
  cudaFree(t_d);
  cudaFree(h_d);
  cudaFree(dt_d);
  cudaFree(rank_d);

	/* timer cleanup */
	CUDA_CALL(cudaEventDestroy(kstart));
	CUDA_CALL(cudaEventDestroy(kstop));
}
