#define _USE_MATH_DEFINES
#define TAU 0.001
#define PAR
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
    vector<vector<double> > base;
    int Nt;
    int N[3];
    vector<vector<double> > to_send;
    vector<vector<double> > to_recv;
    Custom_vector(int nt, int* n) {
        Nt = nt;
        copy(n, n+3, N);
        base = vector<vector<double> >(nt, vector<double>(n[0]*n[1]*n[2]));
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
        return base[t][z+y*N[1]+x*N[0]*N[0]];
    }
    double get_normal(int t, int x, int y, int z) {
        return base[t][z+y*N[1]+x*N[0]*N[0]];
    }
    void set(int t, int x, int y, int z, double value) {
        if (x == 0) to_send[0][z+y*N[1]] = value;
        if (x == N[0]-1) to_send[1][z+y*N[1]] = value;
        if (y == 0) to_send[2][z+x*N[0]] = value;
        if (y == N[1]-1) to_send[3][z+x*N[0]] = value;
        if (z == 0) to_send[4][y+x*N[0]] = value;
        if (z == N[2]-1) to_send[5][y+x*N[0]] = value;
        base[t][z+y*N[1]+x*N[0]*N[0]] = value;
    }

};

struct Counter {
    int N[3];
    double L[3], h[3], base_vals[3], a_t;
    Custom_vector grid;
    vector<double> shared_x;
    Counter(int nt, int* n, double* l, int* bv) {
        for (int i = 0; i<3; i++) {
            N[i] = n[i];
            L[i] = l[i];
            h[i] = l[i]/n[i];
            bv[i] = bv[i]*h[i];
        }
        a_t = M_PI*sqrt(9/(L[0]*L[0]) + 4/(L[1]*L[1]) + 4/(L[2]*L[2]));
        grid = Custom_vector(nt, N);
    }

    double Delta(int i, int j, int k, int n_t) {
        return  (grid.get(n_t, i-1, j, k) - 2*grid.get(n_t, i, j, k)+ grid.get(n_t, i+1, j, k))/(h[0]*h[0]) +
                (grid.get(n_t, i, j-1, k) - 2*grid.get(n_t, i, j, k)+ grid.get(n_t, i, j+1, k))/(h[1]*h[1]) +
                (grid.get(n_t, i, j, k-1) - 2*grid.get(n_t, i, j, k)+ grid.get(n_t, i, j, k+1))/(h[2]*h[2]);
    }

    double Count_u_from_grid (int i, int j, int k, int n_t) {
        double delta = Delta(i, j, k, n_t-1);
        return delta*TAU*TAU + 2*grid.get(n_t-1, i, j, k) - grid.get(n_t-1, i, j, k); 
    }

    double Count_u_analitic (double x, double y, double z, int n_t) {
        double t = n_t*TAU;
        return sin(3*(M_PI/L[0])*(x+base_vals[0])) * sin(2*(M_PI/L[1])*(x+base_vals[0])) * sin(2*(M_PI/L[2])*(x+base_vals[0])) * cos(a_t*t + 4*M_PI);
    }
};

int main(int argc, char *argv[]) {
#pragma region preparations
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

    double L[] = {1, 1, 1};
    double T=1, max;    
    int err, numprocs, myid, Nt = 20;
    int dims[] = {0, 0, 0};
    int periods[3], neighbours_m[3], neighbours_p[3], coords[3];
    MPI_Comm COMM_CART;
    if (err = MPI_Init(&argc, &argv)) { 
        cout << "INIT ERROR!" << endl;
        MPI_Abort(MPI_COMM_WORLD, err);
    } 
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs); 
    MPI_Dims_create(numprocs, 3, dims);
    if (err = MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &COMM_CART)) {
        cout << "CART ERROR!" << endl;
        MPI_Abort(MPI_COMM_WORLD, err);
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
    int base_vals[] = {N[0]*coords[0], N[1]*coords[1], N[1]*coords[1]};
    
    double start; 

    if (myid == 0)
        start  = MPI_Wtime();
#pragma endregion

    Counter counter(Nt, N, L, base_vals);
    for (int t = 0; t < 2; t++)
        for (int i = 0; i < N[0]; i++)
            for (int j = 0; j < N[1]; j++)
                for (int k = 0; k < N[2]; k++)
                    if (i == 0 && coords[0] == 0 || i == N[0] - 1 && coords[0] == dims[0]-1) 
                        counter.grid.set(t, i, j, k, 0);
                    else 
                        counter.grid.set(t, i, j, k, counter.Count_u_analitic(i, j, k, t));
    
    MPI_Status status;
    
    for (int t = 2; t < Nt; t++) {
        for (int dim = 0; dim < 3; dim++){
            if (neighbours_m[dim] != MPI_PROC_NULL && neighbours_p[dim] != MPI_PROC_NULL) {
                MPI_Sendrecv(&(counter.grid.to_send[2*dim][0]), counter.grid.to_send[2*dim].size(), MPI_DOUBLE, neighbours_m[dim], 10,  
                             &(counter.grid.to_recv[2*dim+1][0]), counter.grid.to_recv[2*dim+1].size(), MPI_DOUBLE, neighbours_p[dim], 10, COMM_CART, &status);
                MPI_Sendrecv(&(counter.grid.to_send[2*dim+1][0]), counter.grid.to_send[2*dim+1].size(), MPI_DOUBLE, neighbours_p[dim], 11,  
                             &(counter.grid.to_recv[2*dim][0]), counter.grid.to_recv[2*dim].size(), MPI_DOUBLE, neighbours_m[dim], 11, COMM_CART, &status);
            }
            else if (neighbours_m[dim] != MPI_PROC_NULL) {
                MPI_Send(&(counter.grid.to_send[2*dim][0]), counter.grid.to_send[2*dim].size(), MPI_DOUBLE, neighbours_m[dim], 10, COMM_CART); 
                MPI_Recv(&(counter.grid.to_recv[2*dim][0]), counter.grid.to_recv[2*dim].size(), MPI_DOUBLE, neighbours_m[dim], 11, COMM_CART, &status);         
            }
            else if (neighbours_p[dim] != MPI_PROC_NULL) {
                MPI_Recv(&(counter.grid.to_recv[2*dim+1][0]), counter.grid.to_recv[2*dim+1].size(), MPI_DOUBLE, neighbours_p[dim], 10, COMM_CART, &status);    
                MPI_Send(&(counter.grid.to_send[2*dim+1][0]), counter.grid.to_send[2*dim+1].size(), MPI_DOUBLE, neighbours_p[dim], 11, COMM_CART);  
            }
        }
        for (int i = 0; i < N[0]; i++)
            for (int j = 0; j < N[1]; j++)
                for (int k = 0; k < N[2]; k++)
                    if (i == 0 || i == N[0] - 1)
                        counter.grid.set(t, i, j, k, 0);
                    else {
                        counter.grid.set(t, i, j, k, counter.Count_u_from_grid(i, j, k, t));
                    }
    }

    double my_max = 0, overall_max = 0;
    {
        for (int t = 0; t < Nt; t++)
        for (int i = 0; i < N[0]; i++)
            for (int j = 0; j < N[1]; j++)
                for (int k = 0; k < N[2]; k++) {
                        double my = counter.grid.get(t, i, j, k), ideal = counter.Count_u_analitic(i, j, k, t);
                        double diff = my > ideal ? my-ideal : ideal-my;
                        if (diff > my_max)
                            my_max = diff; 
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
