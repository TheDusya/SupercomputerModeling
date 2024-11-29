


#define _USE_MATH_DEFINES
#define T 0.0001
#define L 1.0
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

struct Custom_vector
{
    vector<vector<vector<vector<double>>>> base;
    int N[3];
    vector<vector<double> > to_send;
    vector<vector<double> > to_recv;
    Custom_vector(int* n) {
        copy(n, n+3, N);
        base = vector<vector<vector<vector<double>>> >(ITER, vector<vector<vector<double>>>(n[0], vector<vector<double>>(n[1], vector<double>(n[2]))));
        for (int d = 0; d < 3; d++)
            for (int j = 0; j < 2; j++) {
                to_send.push_back(vector<double>(n[(d+1)%3] * n[(d+2)%3]));
                to_recv.push_back(vector<double>(n[(d+1)%3] * n[(d+2)%3]));
            }
    }
    Custom_vector(){}
    double get(int t, int x, int y, int z) {
        if (x == -1) return to_recv[0][z+y*N[1]];
        if (x == N[0]) return to_recv[1][z+y*N[1]];
        if (y == -1) return to_recv[2][z+x*N[0]];
        if (y == N[1]) return to_recv[3][z+x*N[0]];
        if (z == -1) return to_recv[4][y+x*N[0]];
        if (z == N[2]) return to_recv[5][y+x*N[0]];
        return base[t][x][y][z];
    }
    void set(int t, int x, int y, int z, double value) {
        if (x == 0) to_send[0][z+y*N[1]] = value;
        if (x == N[0]-1) to_send[1][z+y*N[1]] = value;
        if (y == 0) to_send[2][z+x*N[0]] = value;
        if (y == N[1]-1) to_send[3][z+x*N[0]] = value;
        if (z == 0) to_send[4][y+x*N[0]] = value;
        if (z == N[2]-1) to_send[5][y+x*N[0]] = value;
        base[t][x][y][z] = value;
    }

};

struct Counter {
    int N[3];
    double h[3], base_vals[3], a_t;
    Custom_vector grid;
    vector<double> shared_x;
    Counter(int* n, double* l, int* bv) {
        for (int i = 0; i<3; i++) {
            N[i] = n[i];
            h[i] = l[i]/(i == 0? n[i]-1 : n[i]);
            base_vals[i] = bv[i]*h[i];
        }
        a_t = M_PI*sqrt(9/(L*L) + 4/(L*L) + 4/(L*L));
        grid = Custom_vector(N);
    }

    double Delta(int n_t, int i, int j, int k) {

        return  (grid.get(n_t, i-1, j, k) - 2*grid.get(n_t, i, j, k)+ grid.get(n_t, i+1, j, k))/(h[0]*h[0]) +
                (grid.get(n_t, i, j-1, k) - 2*grid.get(n_t, i, j, k)+ grid.get(n_t, i, j+1, k))/(h[1]*h[1]) +
                (grid.get(n_t, i, j, k-1) - 2*grid.get(n_t, i, j, k)+ grid.get(n_t, i, j, k+1))/(h[2]*h[2]);
    }

    double Count_u_from_grid(int n_t, int i, int j, int k) {
        double delta = Delta(n_t-1, i, j, k);
        return delta*TAU*TAU + 2*grid.get(n_t-1, i, j, k) - grid.get(n_t-2, i, j, k); 
    }

    double Count_u_analitic(int n_t, int i, int j, int k) {
        double real_t = (n_t/ITER)*T;
        return sin(3*(M_PI/L)*(i*h[0]+base_vals[0])) * 
               sin(2*(M_PI/L)*(j*h[1]+base_vals[1])) * 
               sin(2*(M_PI/L)*(k*h[2]+base_vals[2])) * cos(a_t*real_t + 4*M_PI);
    }
};

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
    int dims[] = {0, 0, 0};
    int periods[3], neighbours_m[3], neighbours_p[3], coords[3];
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
    int N[] = {old_N/dims[0], old_N/dims[1], old_N/dims[2]};
    double l[] = {L/dims[0], L/dims[1], L/dims[2]};
    int base_vals[] = {N[0]*coords[0], N[1]*coords[1], N[2]*coords[2]};
    
    double start; 
    if (myid == 0)
        start  = MPI_Wtime();

    Counter counter(N, l, base_vals);
    for (int t = 0; t < 2; t++)
        for (int i = 0; i < N[0]; i++)
            for (int j = 0; j < N[1]; j++)
                for (int k = 0; k < N[2]; k++) {
                    if (i == 0 && coords[0] == 0 || i == N[0]-1 && coords[0] == dims[0]-1)  {
                        counter.grid.set(t, i, j, k, 0);
                    }
                    else {
                        counter.grid.set(t, i, j, k, counter.Count_u_analitic(t, i, j, k));
                    }
                }
    MPI_Status status;
    
    for (int t = 2; t < ITER; t++) {
        for (int dim = 0; dim < 3; dim++) {
            MPI_Request requests[4];
            MPI_Status status[4]; 

            if (neighbours_m[dim] != MPI_PROC_NULL && neighbours_p[dim] != MPI_PROC_NULL) {
                MPI_Isend(&(counter.grid.to_send[2 * dim][0]), counter.grid.to_send[2 * dim].size(), MPI_DOUBLE, 
                        neighbours_m[dim], 10, COMM_CART, &requests[0]);
                MPI_Irecv(&(counter.grid.to_recv[2 * dim + 1][0]), counter.grid.to_recv[2 * dim + 1].size(), MPI_DOUBLE, 
                        neighbours_p[dim], 10, COMM_CART, &requests[1]);

                MPI_Waitall(2, requests, status);

                MPI_Isend(&(counter.grid.to_send[2 * dim + 1][0]), counter.grid.to_send[2 * dim + 1].size(), MPI_DOUBLE, 
                        neighbours_p[dim], 11, COMM_CART, &requests[2]);
                MPI_Irecv(&(counter.grid.to_recv[2 * dim][0]), counter.grid.to_recv[2 * dim].size(), MPI_DOUBLE, 
                        neighbours_m[dim], 11, COMM_CART, &requests[3]);

                MPI_Waitall(2, &requests[2], &status[2]);
            } 
            else if (neighbours_m[dim] != MPI_PROC_NULL) {
                MPI_Isend(&(counter.grid.to_send[2 * dim][0]), counter.grid.to_send[2 * dim].size(), MPI_DOUBLE, 
                        neighbours_m[dim], 10, COMM_CART, &requests[0]);
                MPI_Irecv(&(counter.grid.to_recv[2 * dim][0]), counter.grid.to_recv[2 * dim].size(), MPI_DOUBLE, 
                        neighbours_m[dim], 11, COMM_CART, &requests[1]);

                MPI_Waitall(2, requests, status);
            } 
            else if (neighbours_p[dim] != MPI_PROC_NULL) {
                MPI_Irecv(&(counter.grid.to_recv[2 * dim + 1][0]), counter.grid.to_recv[2 * dim + 1].size(), MPI_DOUBLE, 
                        neighbours_p[dim], 10, COMM_CART, &requests[0]);
                
                MPI_Wait(&requests[0], &status[0]);
                MPI_Isend(&(counter.grid.to_send[2 * dim + 1][0]), counter.grid.to_send[2 * dim + 1].size(), MPI_DOUBLE, 
                        neighbours_p[dim], 11, COMM_CART, &requests[1]);
                
                MPI_Wait(&requests[1], &status[1]);
            }
        }

        for (int i = 0; i < N[0]; i++)
            for (int j = 0; j < N[1]; j++)
                for (int k = 0; k < N[2]; k++)
                    if (i == 0 && coords[0] == 0 || i == N[0]-1 && coords[0] == dims[0]-1) 
                        counter.grid.set(t, i, j, k, 0);
                    else {
                        counter.grid.set(t, i, j, k, counter.Count_u_from_grid(t, i, j, k));
                    }
    }

    double my_max = 0, overall_max = 0;
    {
        for (int t = 0; t < ITER; t++)
            for (int i = 1; i < N[0]-1; i++)
                for (int j = 0; j < N[1]; j++)
                    for (int k = 0; k < N[2]; k++) {
                            double my = counter.grid.get(t, i, j, k);
                            double ideal = counter.Count_u_analitic(t, i, j, k);
                            double diff = my > ideal ? my-ideal : ideal-my;
                            if (diff > my_max){
                                my_max = diff; 
                            }
                    }
        MPI_Reduce(&my_max, &overall_max, 1, MPI_DOUBLE, MPI_MAX, 0, COMM_CART);

    }
    if (myid == 0) {
        double end = MPI_Wtime();
        cout << "Work took " << end-start << " seconds, max error - " << scientific << overall_max << endl;
    }
    MPI_Finalize();
    return 0;
}
