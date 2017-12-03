/******************************************************************************
* FILE: lake_mpi.cu
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
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

#define _USE_MATH_DEFINES

#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0

#define MAX_PSZ 10
#define TSCALE 1.0
#define VSQR 0.1

void init(double *u, double *pebbles, int n);
void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);
int tpdt(double *t, double dt, double end_time);
void print_heatmap(const char *filename, double *u, int n, double h);
void print_heatmap_custom(const char *filename, double *u, int n, double h);
void init_pebbles(double *p, int pn, int n);

void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time);

extern void run_gpu(double *u, double *u0, double *u1, double *pebbles,
                    int n, double h, double end_time, int nthreads, int rank);

int main(int argc, char *argv[]){

  if(argc != 5){
    printf("Usage: %s npoints npebs time_finish nthreads \n",argv[0]);
    return 0;
  }

  int     npoints   = atoi(argv[1]);
  int     npebs     = atoi(argv[2]);
  double  end_time  = (double)atof(argv[3]);
  int     nthreads  = atoi(argv[4]);
  int 	  narea	    = npoints * npoints;
  int     rank, size;

  double *u_i0, *u_i1;
  double *u_cpu, *u_gpu, *pebs;
  double h;

  double elapsed_cpu, elapsed_gpu;
  struct timeval cpu_start, cpu_end, gpu_start, gpu_end;

  u_i0 = (double*)malloc(sizeof(double) * narea);
  u_i1 = (double*)malloc(sizeof(double) * narea);
  pebs = (double*)malloc(sizeof(double) * narea);

  u_cpu = (double*)malloc(sizeof(double) * narea);
  u_gpu = (double*)malloc(sizeof(double) * narea);

  memset(u_gpu, 0, sizeof(double)*narea);

  //initializing MPI here
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  //only allow 4 processsors
  if(size != 4){
    exit(1);
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0){
    printf("Running %s with (%d x %d) grid, until %f, with %d threads\n", argv[0], npoints, npoints,
                                                                          end_time, nthreads);
  }

  h = (XMAX - XMIN)/npoints;

  //construct pebble locations at root processor for the grid
  if (rank == 0){
    init_pebbles(pebs, npebs, npoints);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Broadcast pebble information to all other processes
  MPI_Bcast(pebs, narea, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  //Initialize current two timestamps
  init(u_i0, pebs, npoints);
  init(u_i1, pebs, npoints);

  //print the initial state of the grid (using CPU)
  if (rank == 0){

    print_heatmap("lake_i.dat", u_i0, npoints, h);
    gettimeofday(&cpu_start, NULL);
    run_cpu(u_cpu, u_i0, u_i1, pebs, npoints, h, end_time);
    gettimeofday(&cpu_end, NULL);

    elapsed_cpu = ((cpu_end.tv_sec + cpu_end.tv_usec * 1e-6)-(
                  cpu_start.tv_sec + cpu_start.tv_usec * 1e-6));
    printf("CPU took %f seconds\n", elapsed_cpu);
    print_heatmap("lake_f.dat", u_cpu, npoints, h);
  }

  // Wait for processor to complete cpu calculation
  MPI_Barrier(MPI_COMM_WORLD);

  gettimeofday(&gpu_start, NULL);

  //run GPU code to compute the ripples in the grid
  run_gpu(u_gpu, u_i0, u_i1, pebs, npoints, h, end_time, nthreads, rank);
  gettimeofday(&gpu_end, NULL);
  elapsed_gpu = ((gpu_end.tv_sec + gpu_end.tv_usec * 1e-6)-(
                  gpu_start.tv_sec + gpu_start.tv_usec * 1e-6));

  MPI_Barrier(MPI_COMM_WORLD);

  //root processors prints out GPU timing
  if(rank == 0){
    printf("GPU took %f seconds\n", elapsed_gpu);
  }

  //print out different .dat files for the quadrant computed at each node
  char filename[13];
  sprintf(filename,"lake_f_%d.dat", rank);

  print_heatmap(filename, u_gpu, npoints, h);

  //free resources
  free(u_i0);
  free(u_i1);
  free(pebs);
  free(u_cpu);
  free(u_gpu);

  //graceful exit from MPI
  MPI_Finalize();
  return 0;
}

// simulates the state of the grid after the given time, using a 13-point stencil function
void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time){

  double *un, *uc, *uo;
  double t, dt;

  un = (double*)malloc(sizeof(double) * n * n);
  uc = (double*)malloc(sizeof(double) * n * n);
  uo = (double*)malloc(sizeof(double) * n * n);

  memcpy(uo, u0, sizeof(double) * n * n);
  memcpy(uc, u1, sizeof(double) * n * n);

  t = 0.;
  dt = h / 2.;

  while(1){

    evolve(un, uc, uo, pebbles, n, h, dt, t);
    //save most recent two time-stamps into uo and uc
    memcpy(uo, uc, sizeof(double) * n * n);
    memcpy(uc, un, sizeof(double) * n * n);

    if(!tpdt(&t,dt,end_time)) break;
  }

  memcpy(u, un, sizeof(double) * n * n);
}

//initialize the location of the pebbles in the grid
void init_pebbles(double *p, int pn, int n){

  int i, j, k, idx;
  int sz;

  srand( time(NULL) );
  memset(p, 0, sizeof(double) * n * n);

  for( k = 0; k < pn ; k++ ){

    i = rand() % (n - 4) + 2;
    j = rand() % (n - 4) + 2;
    sz = rand() % MAX_PSZ;
    idx = j + i * n;
    p[idx] = (double) sz;
  }

}


// //host function to compute function f()
__host__ double f(double p, double t){
  return -expf(-TSCALE * t) * p;
}

//moves time forward by dt and checks if the time is under the given time limit
int tpdt(double *t, double dt, double tf){

  if((*t) + dt > tf) return 0;
  (*t) = (*t) + dt;
  return 1;
}

//initializes the grid along with pebble location
void init(double *u, double *pebbles, int n){

  int i, j, idx;

  for(i = 0; i < n ; i++){
    for(j = 0; j < n ; j++){
      idx = j + i * n;
      u[idx] = f(pebbles[idx], 0.0);
    }
  }
}

//updates the grid state from time t to time t+dt
void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t){

  int i, j, idx;

  for( i = 0; i < n; i++){
    for( j = 0; j < n; j++){

      idx = j + i * n;
      //values at lake edge points are set to zero
      if( i == 0 || i == n - 1 || j == 0 || j == n - 1
          || i == n - 2 || i == 1 || j == n - 2 || j == 1){

        un[idx] = 0.;
      }
      else{
        //compute the 13-point stencil function for every grid point
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *
                  ((  uc[idx-1] // WEST
                      + uc[idx+1] // EAST
                      + uc[idx + n] // SOUTH
                      + uc[idx - n] // NORTH
                     + 0.25*( uc[idx - n - 1 ] // NORTHWEST
                            + uc[idx - n + 1 ] // NORTHEAST
                            + uc[idx + n - 1 ] // SOUTHWEST
                            + uc[idx + n + 1 ] // SOUTHEAST
                            )
                    + 0.125*( uc[idx - 2 ]  // WESTWEST
                            + uc[idx + 2 ] // EASTEAST
                            + uc[idx - 2 * n ] // NORTHNORTH
                            + uc[idx + 2 * n ] // SOUTHSOUTH
                            )
                    - 6 * uc[idx])/(h * h) + f(pebbles[idx],t));
      }
    }
  }
}

//print grid values to a file
void print_heatmap(const char *filename, double *u, int n, double h){

  int i, j, idx;

  FILE *fp = fopen(filename, "w");

  for( i = 0; i < n; i++ ){
    for( j = 0; j < n; j++ ){

      idx = j + i * n;

      fprintf(fp, "%f %f %f\n", i*h, j*h, u[idx]);
    }
  }

  fclose(fp);
}


/*void print_heatmap_custom(const char *filename, double *u, int n, double h){

  int i, j, idx;

  FILE *fp = fopen(filename, "w");

  for( i = 0; i < n; i++ ){
    for( j = 0; j < n; j++ ){

      idx = j + i * n;
      // idx = j*n + i;
      fprintf(fp, "%f %f %f\n", j*h, i*h, u[idx]);
    }
  }

  fclose(fp);
} */


