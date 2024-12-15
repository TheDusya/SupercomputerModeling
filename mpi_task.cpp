#define _USE_MATH_DEFINES
#define T 0.0001
#define L 1.0
//#define L M_PI
#define TAU T/20
#define ITER 20.0

#include <iostream>
#include <string>
#include <math.h>
#include <algorithm>
#include <iterator>
#include "mpi.h"
#include <vector>
using namespace std;

double get(double* base, double** to_recv, int N[3], int sh, int x, int y, int z) {
    int big_N = *(max_element(N, N+3));
    if (x == -1) return to_recv[0+sh][z+y*big_N];
    if (x == N[0]) return to_recv[1+sh][z+y*big_N];
    if (y == -1) return to_recv[2+sh][z+x*big_N];
    if (y == N[1]) return to_recv[3+sh][z+x*big_N];
    if (z == -1) return to_recv[4+sh][y+x*big_N];
    if (z == N[2]) return to_recv[5+sh][y+x*big_N];
    return base[z+y*big_N+x*big_N*big_N];
}
void set(double* base, double** to_send, int N[3], int sh, int x, int y, int z, double value) {
    int big_N = *(max_element(N, N+3));
    if (x == 0) to_send[0+sh][z+y*big_N] = value;
    if (x == N[0]-1) to_send[1+sh][z+y*big_N] = value;
    if (y == 0) to_send[2+sh][z+x*big_N] = value;
    if (y == N[1]-1) to_send[3+sh][z+x*big_N] = value;
    if (z == 0) to_send[4+sh][y+x*big_N] = value;
    if (z == N[2]-1) to_send[5+sh][y+x*big_N] = value;
    base[z+y*big_N+x*big_N*big_N] = value;
}

double Delta(double* base, double** to_recv, int N[3], double h[3], int sh, int x, int y, int z) {
        return  (get(base, to_recv, N, sh, x-1, y, z) - 2*get(base, to_recv, N, sh, x, y, z)+ get(base, to_recv, N, sh, x+1, y, z))/(h[0]*h[0]) +
                (get(base, to_recv, N, sh, x, y-1, z) - 2*get(base, to_recv, N, sh, x, y, z)+ get(base, to_recv, N, sh, x, y+1, z))/(h[1]*h[1]) +
                (get(base, to_recv, N, sh, x, y, z-1) - 2*get(base, to_recv, N, sh, x, y, z)+ get(base, to_recv, N, sh, x, y, z+1))/(h[2]*h[2]);
}

double Count_from_grid(double** base, double** to_recv, int N[3], double h[3], int t, int x, int y, int z) {
    double delta = Delta(base[(t-1)%3], to_recv, N, h, 6*((t-1)%3), x, y, z);
    return delta*TAU*TAU + 2*get(base[(t-1)%3], to_recv, N, 6*((t-1)%3), x, y, z) - get(base[(t-2)%3], to_recv, N, 6*((t-2)%3), x, y, z); 
}

double Count_analitic(double a_t, double h[3], double base_vals[3], int t, int x, int y, int z) {
    double real_t = (t/ITER)*T;
    return  sin(3*(M_PI/L)*(x*h[0]+base_vals[0])) * 
            sin(2*(M_PI/L)*(y*h[1]+base_vals[1])) * 
            sin(2*(M_PI/L)*(z*h[2]+base_vals[2])) * cos(a_t*real_t + 4*M_PI);
}

int main(int argc, char *argv[]) {
    int old_N;
    if (argc < 2)
        old_N  = 50;
    else
        try {
            old_N  = atoi(argv[1]);
        }
        catch (exception e) {
            cout << "Bad arguments!" << endl;
            return 1;
        }

    int err, numprocs, myid;
    int dims[] = {0, 0, 0}, periods[] = {0, 1, 1};
    int neighbours_m[3], neighbours_p[3], coords[3];
    MPI_Comm COMM_CART;
    if (MPI_Init(&argc, &argv)) { 
        cout << "INIT ERROR!" << endl;
        exit(1);
    } 
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs); 
    MPI_Dims_create(numprocs, 3, dims);
    if (MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &COMM_CART)) {
        cout << "CART ERROR!" << endl;
        exit(1);
    }
    MPI_Comm_rank(COMM_CART, &myid);
    if (old_N % dims[0] != 0 || old_N % dims[1] != 0 || old_N % dims[2] != 0) {
        cout << "I have no idea what to do :(" << endl;
        exit(1);
    }
    for (int dim = 0; dim < 3; dim++)
        MPI_Cart_shift(COMM_CART, dim, 1, &neighbours_m[dim], &neighbours_p[dim]);
    MPI_Cart_coords (COMM_CART, myid, 3, coords);
    int N[3];
    double results[(int)ITER], l[3], h[3], base_vals[3], a_t = M_PI*sqrt(9/(L*L) + 4/(L*L) + 4/(L*L));
    for (int i = 0; i<3; i++) {
        N[i] = (int) ceil(old_N / (double) dims[i]);
        h[i] = L/(old_N-1);
        base_vals[i] = N[i]*coords[i]*h[i];
    }
    int big_N = *(max_element(N, N+3));
    int buf_size = big_N*big_N;

    double** base = (double**)malloc(3 * sizeof(double*));
    double** to_send = (double**)malloc(18 * sizeof(double*));
    double** to_recv = (double**)malloc(18 * sizeof(double*));
    for (int i = 0; i < 3; i++) {
        base[i] = (double*)malloc(big_N*big_N*big_N * sizeof(double));
    }
    for (int i = 0; i < 18; i++) {
        to_send[i] = (double*)malloc(buf_size * sizeof(double));
        to_recv[i] = (double*)malloc(buf_size * sizeof(double));
    }

    double start; 
    if (myid == 0)
        start  = MPI_Wtime();
        
    for (int t = 0; t < ITER; t++) {
        int sh = 6*(t%3);
        for (int i = 0; i < N[0]; i++)
            for (int j = 0; j < N[1]; j++)
                for (int k = 0; k < N[2]; k++)
                    if (i == 0 && coords[0] == 0 || i == N[0]-1 && coords[0] == dims[0]-1) 
                        set(base[t%3], to_send, N, sh, i, j, k, 0);
                    else if (t < 2) 
                        set(base[t%3], to_send, N, sh, i, j, k, Count_analitic(a_t, h, base_vals, t, i, j, k));
                    else 
                        set(base[t%3], to_send, N, sh, i, j, k, Count_from_grid(base, to_recv, N, h, t, i, j, k));
                        
        for (int dim = 0; dim < 3; dim++) {
            MPI_Request requests[4];
            MPI_Status status[4]; 
            if (neighbours_m[dim] != MPI_PROC_NULL && neighbours_p[dim] != MPI_PROC_NULL) {
                MPI_Isend(to_send[2*dim + sh], buf_size, MPI_DOUBLE, neighbours_m[dim], 10, COMM_CART, &requests[0]);
                MPI_Irecv(to_recv[2*dim+1 + sh], buf_size, MPI_DOUBLE, neighbours_p[dim], 10, COMM_CART, &requests[1]);
                MPI_Waitall(2, requests, status);
                
                MPI_Isend(to_send[2*dim+1 + sh], buf_size, MPI_DOUBLE, neighbours_p[dim], 11, COMM_CART, &requests[2]);
                MPI_Irecv(to_recv[2*dim + sh], buf_size, MPI_DOUBLE, neighbours_m[dim], 11, COMM_CART, &requests[3]);
                MPI_Waitall(2, &requests[2], &status[2]);
            } 
            else if (neighbours_m[dim] != MPI_PROC_NULL) {
                MPI_Isend(to_send[2*dim + sh], buf_size, MPI_DOUBLE, neighbours_m[dim], 10, COMM_CART, &requests[0]);
                MPI_Irecv(to_recv[2*dim + sh], buf_size, MPI_DOUBLE, neighbours_m[dim], 11, COMM_CART, &requests[1]);
                MPI_Waitall(2, requests, status);
            } 
            else if (neighbours_p[dim] != MPI_PROC_NULL) {
                MPI_Irecv(to_recv[2*dim+1 + sh], buf_size, MPI_DOUBLE, neighbours_p[dim], 10, COMM_CART, &requests[0]);
                MPI_Wait(&requests[0], &status[0]);

                MPI_Isend(to_send[2*dim+1 + sh], buf_size, MPI_DOUBLE, neighbours_p[dim], 11, COMM_CART, &requests[1]);
                MPI_Wait(&requests[1], &status[1]);
            }
        } 
        double my_max = 0, overall_max = 0;
        if (t > 2) {
            for (int i = 1; i < N[0]-1; i++)
                for (int j = 0; j < N[1]; j++)
                    for (int k = 0; k < N[2]; k++) {
                        double my = base[t%3][k+j*big_N+i*big_N*big_N];
                        double ideal = Count_analitic(a_t, h, base_vals, t, i, j, k);
                        double diff = my > ideal ? my-ideal : ideal-my;
                        if (diff > my_max)
                            my_max = diff;
                    }
            
            MPI_Reduce(&my_max, &overall_max, 1, MPI_DOUBLE, MPI_MAX, 0, COMM_CART);
        }
        if (myid == 0)
            results[t] = overall_max;
    }

    if (myid == 0) {
        cout.precision(3); 
        double end = MPI_Wtime();
        cout << "Work took " << end-start << " seconds, max error - " << scientific << *max_element(results, results+(int)ITER) << endl;
    }
    
    if (to_send != NULL) {
        for (int i = 0; i < 6; i++) {
            free(to_send[i]);
        }
        free(to_send);
    }
    if (to_recv != NULL) {
        for (int i = 0; i < 6; i++) {
            free(to_recv[i]);
        }
        free(to_recv);
    }
    if (base != NULL) {
        for (int i = 0; i < 3; i++) {
            free(base[i]);
        }
        free(base);
    }
    MPI_Finalize();
    return 0;
}
