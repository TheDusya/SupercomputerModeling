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
#include "omp.h"
#include <vector>
using namespace std;

struct Custom_vector
{
    vector<vector<double> > base;
    int N;
    Custom_vector(int n) {
        N = n;
        base = vector<vector<double> >(ITER, vector<double>(n * n * n));
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

    Counter(int n, double lx, double ly, double lz) {
        N = n;
        Lx = lx;
        Ly = ly;
        Lz = lz;
        hx = Lx/N;
        hy = Ly/N;
        hz = Lz/N;
        a_t = M_PI*sqrt(9.0/(Lx*Lx) + 4.0/(Ly*Ly) + 4.0/(Lz*Lz));
        grid = Custom_vector(n+1);
    }

    double Delta(int n_t, int i, int j, int k) {
        int jm1 = j == 0 ? N : j-1;
        int jp1 = j == N ? 0 : j+1;
        int km1 = k == 0 ? N : k-1;
        int kp1 = k == N ? 0 : k+1;
        return  (double)(grid.get(n_t, i-1, j, k) - 2*grid.get(n_t, i, j, k)+ grid.get(n_t, i+1, j, k))/(hx*hx) +
                (double)(grid.get(n_t, i, jm1, k) - 2*grid.get(n_t, i, j, k)+ grid.get(n_t, i, jp1, k))/(hy*hy) +
                (double)(grid.get(n_t, i, j, km1) - 2*grid.get(n_t, i, j, k)+ grid.get(n_t, i, j, kp1))/(hz*hz);
    }

    double Count_u_from_grid (int n_t, int i, int j, int k) {
        double delta = Delta(n_t-1, i, j, k);
        return delta*TAU*TAU + 2*grid.get(n_t-1, i, j, k) - grid.get(n_t-2, i, j, k); 
    }

    double Count_u_analitic (int n_t, int i, int j, int k) {
        double real_t = (n_t/ITER)*T;
        return sin(3*(M_PI/Lx)*i*hx) * sin(2*(M_PI/Ly)*j*hy) * sin(2*(M_PI/Lz)*k*hz) * cos(a_t*real_t + 4*M_PI);
    }
};

int main(int argc, char const *argv[]) {
    int N;
    double Lx=L, Ly=L, Lz=L; 
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
    double start, end; 
    Counter counter(N, Lx, Ly, Lz);
    double results[(int)ITER] = { 0 }; 
    start = omp_get_wtime(); 

    for (int t = 0; t < ITER; t++) {
        #pragma omp parallel
        #pragma omp for collapse(3)
        for (int i = 0; i <= N; i++)
            for (int j = 0; j <= N; j++)
                for (int k = 0; k <= N; k++)
                    if (i == 0 || i == N)
                        counter.grid.set(t, i, j, k, 0);
                    else if (t < 2)
                        counter.grid.set(t, i, j, k, counter.Count_u_analitic(t, i, j, k));
                    else
                        counter.grid.set(t, i, j, k, counter.Count_u_from_grid(t, i, j, k));
        #pragma omp barrier
        double overall_max = 0;
        if (t > 2) {
            #pragma omp parallel
            #pragma omp for collapse(3) reduction(max:overall_max)
                for (int i = 1; i <= N-1; i++)
                    for (int j = 0; j <= N; j++)
                        for (int k = 0; k <= N; k++) {
                            double my = counter.grid.get(t, i, j, k), ideal = counter.Count_u_analitic(t, i, j, k);
                            double diff = my > ideal ? my-ideal : ideal-my;
                            if (diff > overall_max)
                                overall_max = diff; 
                        }
            results[t] = overall_max;
        }
    }

    end = omp_get_wtime(); 
    cout.precision(3); 
    cout << "Work took " << end-start << " seconds, max error - " << scientific << *max_element(results, results+(int)ITER) << endl;
    return 0;
}