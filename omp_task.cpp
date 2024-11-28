#define _USE_MATH_DEFINES
#define TAU 0.0001
#define PAR

#include <iostream>
#include <string>
#include <math.h>
#include <algorithm>
#include <iterator>
#include "omp.h"
#include <vector>
using namespace std;

struct Custom_vector
{
    vector<vector<double> > base;
    int Nt, N;
    Custom_vector(int nt, int n) {
        Nt = nt;
        N = n;
        base = vector<vector<double> >(nt, vector<double>(n * n * n));
    }
    Custom_vector(){}
    double get(int t, int x, int y, int z) {
        return base[t][z+y*N+x*N*N];
    }
    void set(int t, int x, int y, int z, double value) {
        base[t][z+y*N+x*N*N] = value;
    }
};

struct Counter {
    int N;
    double Lx, Ly, Lz, hx, hy, hz, a_t;
    Custom_vector grid;

    Counter(int nt, int n, double lx, double ly, double lz) {
        N = n;
        Lx = lx;
        Ly = ly;
        Lz = lz;
        hx = Lx/N;
        hy = Ly/N;
        hz = Lz/N;
        a_t = M_PI*sqrt(9/(Lx*Lx) + 4/(Ly*Ly) + 4/(Lz*Lz));
        grid = Custom_vector(nt, n);
    }

    double Delta(int i, int j, int k, int n_t) {
        int jm1 = j == 0 ? N-1 : j-1;
        int jp1 = j == N-1 ? 0 : j+1;
        int km1 = k == 0 ? N-1 : k-1;
        int kp1 = k == N-1 ? 0 : k+1;
        return  (grid.get(n_t, i-1, j, k) - 2*grid.get(n_t, i, j, k)+ grid.get(n_t, i+1, j, k))/(hx*hx) +
                (grid.get(n_t, i, jm1, k) - 2*grid.get(n_t, i, j, k)+ grid.get(n_t, i, jp1, k))/(hy*hy) +
                (grid.get(n_t, i, j, km1) - 2*grid.get(n_t, i, j, k)+ grid.get(n_t, i, j, kp1))/(hz*hz);
    }

    double Count_u_from_grid (int i, int j, int k, int n_t) {
        double delta = Delta(i, j, k, n_t-1);
        return delta*TAU*TAU + 2*grid.get(n_t-1, i, j, k) - grid.get(n_t-1, i, j, k); 
    }

    double Count_u_analitic (double x, double y, double z, int n_t) {
        double t = n_t*TAU;
        return sin(3*(M_PI/Lx)*x) * sin(2*(M_PI/Ly)*y) * sin(2*(M_PI/Lz)*z) * cos(a_t*t + 4*M_PI);
    }
};

int main(int argc, char const *argv[]) {
    int N;
    double Lx=1, Ly=1, Lz=1, T=1, max;    
    if (argc < 2)
        N  = 50;
    else {
        try {
            N  = atoi(argv[1]);
        }
        catch (exception e) {
            cout << "Bad arguments!" << endl;
            return 1;
        }
    }
    int Nt = 20;
    double start, end; 
    start = omp_get_wtime(); 
    Counter counter(Nt, N, Lx, Ly, Lz);

    #pragma omp parallel
    #pragma omp for collapse(4)
    for (int t = 0; t < 2; t++)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < N; k++)
                    if (i == 0 || i == N - 1) 
                        counter.grid.set(t, i, j, k, 0);
                    else 
                        counter.grid.set(t, i, j, k, counter.Count_u_analitic(i, j, k, t));

    #pragma omp barrier
    
    for (int t = 2; t < Nt; t++) {
        #pragma omp parallel
        #pragma omp for collapse(3)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < N; k++)
                    if (i == 0 || i == N - 1)
                        counter.grid.set(t, i, j, k, 0);
                    else {
                        counter.grid.set(t, i, j, k, counter.Count_u_from_grid(i, j, k, t));
                    }
        #pragma omp barrier
    }

    double overall_max = 0;
    #pragma omp parallel
    {
        #pragma omp for collapse(4) reduction(max:overall_max)
        for (int t = 0; t < Nt; t++)
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    for (int k = 0; k < N; k++) {
                        double my = counter.grid.get(t, i, j, k), ideal = counter.Count_u_analitic(i, j, k, t);
                        double diff = my > ideal ? my-ideal : ideal-my;
                        if (diff > overall_max)
                            overall_max = diff; 
                    }
    }
    end = omp_get_wtime(); 
    cout << "Work took " << end-start << " seconds, max error - " << scientific << overall_max << endl;
    system("pause");
    return 0;
}