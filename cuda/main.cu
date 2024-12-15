#define _USE_MATH_DEFINES
#define T 0.0001
#define L 1.0
//#define L M_PI
#define TAU T/20
#define ITER 20.0
#define arr_at(arr, t, base_size) (double*)arr+((t)%3)*base_size
#include <iostream>
#include <string>
#include <math.h>
#include <algorithm>
#include <iterator>
#include "mpi.h"
#include <vector>
using namespace std;

struct Data
{
    int* N;
    int big_N;
    double* h;
    double* base_vals;
    double a_t;
    int base_size;
    int buf_size;
};

void with_check(cudaError_t code, string message = "GPU error")
{
    if (code != cudaSuccess) {
        cout<<message<<": "<<cudaGetErrorString(code) <<endl;
        exit(1);
    }
}

template<class pointer>
__device__ pointer my_max_element(pointer first, pointer last)
{
    if (first == last)
        return last;
 
    pointer largest = first;
 
    while (++first != last)
        if (*largest < *first)
            largest = first;
 
    return largest;
}

__device__ double get(double* base, double* to_recv, Data d, int x, int y, int z) { 
    if (x == -1) return to_recv[0*d.buf_size+z+y*d.big_N];
    if (x == d.N[0]) return to_recv[1*d.buf_size+z+y*d.big_N];
    if (y == -1) return to_recv[2*d.buf_size+z+x*d.big_N];
    if (y == d.N[1]) return to_recv[3*d.buf_size+z+x*d.big_N];
    if (z == -1) return to_recv[4*d.buf_size+y+x*d.big_N];
    if (z == d.N[2]) return to_recv[5*d.buf_size+y+x*d.big_N];
    return base[z+y*d.big_N+x*d.big_N*d.big_N];
}
__device__ void set(double* base, double* to_send, Data d, int x, int y, int z, double value) {
    if (x == 0) to_send[0*d.buf_size+z+y*d.big_N] = value;
    if (x == d.N[0]-1) to_send[1*d.buf_size+z+y*d.big_N] = value;
    if (y == 0) to_send[2*d.buf_size+z+x*d.big_N] = value;
    if (y == d.N[1]-1) to_send[3*d.buf_size+z+x*d.big_N] = value;
    if (z == 0) to_send[4*d.buf_size+y+x*d.big_N] = value;
    if (z == d.N[2]-1) to_send[5*d.buf_size+y+x*d.big_N] = value;
    base[z+y*d.big_N+x*d.big_N*d.big_N] = value;
}

__device__ double Delta(double* base, double* to_recv, Data d, int x, int y, int z) {
        return (get(base, to_recv, d, x-1, y, z) - 2*get(base, to_recv, d, x, y, z)+ get(base, to_recv, d, x+1, y, z))/(d.h[0]*d.h[0]) +
                (get(base, to_recv, d, x, y-1, z) - 2*get(base, to_recv, d, x, y, z)+ get(base, to_recv, d, x, y+1, z))/(d.h[1]*d.h[1]) +
                (get(base, to_recv, d, x, y, z-1) - 2*get(base, to_recv, d, x, y, z)+ get(base, to_recv, d, x, y, z+1))/(d.h[2]*d.h[2]);
}

__device__ double Count_from_grid(double* basem1, double* basem2, double* to_recv1, double* to_recv2, Data d, int x, int y, int z) {
    double delta = Delta(basem1, to_recv1, d, x, y, z);
    return delta*TAU*TAU + 2*get(basem1, to_recv1, d, x, y, z) - get(basem2, to_recv2, d, x, y, z); 
}

__device__ double Count_analitic(Data d, int t, int x, int y, int z) {
    double real_t = (t/ITER)*T;
    return  sin(3*(M_PI/L)*(x*d.h[0]+d.base_vals[0])) * 
            sin(2*(M_PI/L)*(y*d.h[1]+d.base_vals[1])) * 
            sin(2*(M_PI/L)*(z*d.h[2]+d.base_vals[2])) * cos(d.a_t*real_t + 4*M_PI);
}

__global__ void cuda_set(double* big_base, double* to_send, double* big_to_recv, Data d, int t, bool is_beginning, bool is_end)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    if (i >= d.N[0] || j >= d.N[1] || k >= d.N[2])
        return;
    if (i == 0 && is_beginning || i == d.N[0]-1 && is_end)
        set(arr_at(big_base, t, d.base_size), to_send, d, i, j, k, 0);
    else if (t < 2)
        set(arr_at(big_base, t, d.base_size), to_send, d, i, j, k, Count_analitic(d, t, i, j, k));
    else{
        double e = Count_from_grid(arr_at(big_base, t-1, d.base_size), arr_at(big_base, t-2, d.base_size), 
                                                    arr_at(big_to_recv, t-1, d.buf_size*6), arr_at(big_to_recv, t-2, d.buf_size*6), d, i, j, k);
        set(arr_at(big_base, t, d.base_size), to_send, d, i, j, k, e);
    }
}

__device__ double atomicMaxDouble(double* result, double val) {
    unsigned long long* result_to_long = (unsigned long long*)(result+0);
    unsigned long long old_val = *result_to_long;
    unsigned long long new_val = __double_as_longlong(val);
    
    while (new_val > old_val)
        old_val = atomicCAS(result_to_long, old_val, new_val);
    
    return __longlong_as_double(old_val);
}

__global__ void cuda_diff(double* base, Data d, int t, double* result)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    if (i >= d.N[0] || j >= d.N[1] || k >= d.N[2])
        return;
    double ideal = Count_analitic(d, t, i, j, k);
    double my = base[k+j*d.big_N+i*d.big_N*d.big_N];
    double abs = my > ideal ? my-ideal : ideal-my;
    *result = atomicMaxDouble(result, abs);
}

struct Timer {
    double start_time;
    double accumulated_time{0};
    bool running = false;

    void start() {
        if (!running) {
            running = true;
            start_time = MPI_Wtime();
        }
    }

    void stop() {
        if (running) {
            auto end_time = MPI_Wtime();
            accumulated_time += end_time - start_time;
            running = false;
        }
    }

    double get_time() {
        if (running) {
            auto current_time = MPI_Wtime();;
            auto totalTime = accumulated_time + current_time - start_time;
            return totalTime;
        }
        return accumulated_time;
    }
};

int main(int argc, char *argv[]) {
    int old_N;
    Timer common, init, iters, gpu, mpi, finish;
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

    //MPI initialization
    int numprocs, myid;
    int dims[] = {0, 0, 0}, periods[] = {0, 1, 1};
    int neighbours_m[3], neighbours_p[3], coords[3];
    MPI_Comm COMM_CART;
    if (MPI_Init(&argc, &argv)) { 
        cout << "INIT ERROR!" << endl;
        exit(1);
    } 
    common.start();
    init.start();
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs); 
    MPI_Dims_create(numprocs, 3, dims);
    if (MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &COMM_CART)) {
        cout << "CART ERROR!" << endl;
        exit(1);
    }
    MPI_Comm_rank(COMM_CART, &myid);
    for (int dim = 0; dim < 3; dim++)
        MPI_Cart_shift(COMM_CART, dim, 1, &neighbours_m[dim], &neighbours_p[dim]);
    MPI_Cart_coords (COMM_CART, myid, 3, coords);

    //Memory
    int num_cuda_devices;
    bool is_beginning = coords[0] == 0;
    bool is_end = coords[0] == dims[0]-1;
    cudaGetDeviceCount(&num_cuda_devices);
    cudaSetDevice(myid % num_cuda_devices);
    Data data, dev_data;
    
    data.N = (int*)malloc(3 * sizeof(int));
    data.h = (double*)malloc(3 * sizeof(double));
    data.base_vals = (double*)malloc(3 * sizeof(double));
    data.a_t= M_PI*sqrt(9/(L*L) + 4/(L*L) + 4/(L*L));

    with_check(cudaMalloc((void**)&(dev_data.N), 3 * sizeof(int)), "N malloc");
    with_check(cudaMalloc((void**)&(dev_data.h), 3 * sizeof(double)), "h malloc");
    with_check(cudaMalloc((void**)&(dev_data.base_vals), 3 * sizeof(double)), "base_vals malloc");
    dev_data.a_t= M_PI*sqrt(9/(L*L) + 4/(L*L) + 4/(L*L));

    for (int i = 0; i<3; i++) {
        data.N[i] = (int) ceil(old_N / (double) dims[i]);
        data.h[i] = L/(old_N-1);
        data.base_vals[i] = data.N[i]*coords[i]*data.h[i];
    }
    int big_N = *(max_element(data.N, data.N+3));
    int buf_size = big_N*big_N;
    int base_size = big_N*big_N*big_N;
    data.big_N = big_N;
    dev_data.big_N = big_N;
    data.buf_size = buf_size;
    dev_data.buf_size = buf_size;
    data.base_size = base_size;
    dev_data.base_size = base_size;

    dim3 threads(8, 8, 8);
    dim3 blocks(ceil(double(data.N[0])/threads.x), ceil(double(data.N[1])/threads.y), ceil(double(data.N[2])/threads.z));
    gpu.start();
    with_check(cudaMemcpy(dev_data.N, data.N, 3*sizeof(int), cudaMemcpyHostToDevice), "N copy"); 
    with_check(cudaMemcpy(dev_data.h, data.h, 3*sizeof(double), cudaMemcpyHostToDevice), "h copy"); 
    with_check(cudaMemcpy(dev_data.base_vals, data.base_vals, 3*sizeof(double), cudaMemcpyHostToDevice), "base_vals copy");  
    gpu.stop();
    double **base, **to_send, **to_recv, *results, *dev_base, *dev_to_send, *dev_to_recv, *dev_results;

    base = (double**)malloc(base_size * 3 * sizeof(double*));
    to_send = (double**)malloc(18 * sizeof(double*));
    to_recv = (double**)malloc(18 * sizeof(double*));
    with_check(cudaMalloc((void**)&dev_base, base_size * 3 * sizeof(double)), "malloc base");
    with_check(cudaMalloc((void**)&dev_to_send, buf_size * 18 * sizeof(double)), "malloc send");
    with_check(cudaMalloc((void**)&dev_to_recv, buf_size * 18 * sizeof(double)), "malloc recv");

    for (int i = 0; i < 3; i++) {
        base[i] = (double*)malloc(base_size * sizeof(double));
    }
    for (int i = 0; i < 18; i++) {
        to_send[i] = (double*)malloc(buf_size * sizeof(double));
        to_recv[i] = (double*)malloc(buf_size * sizeof(double));
    }
    results = (double*)malloc((int)ITER * sizeof(double));
    with_check(cudaMalloc((void**)&dev_results, (int)ITER * sizeof(double)), "malloc results");
    with_check(cudaMemset(dev_results, 0, (int)ITER * sizeof(double)), "memset results");
    
    //main part
    cudaDeviceSynchronize();
    for (int t = 0; t < ITER; t++) {
        int sh = 6*(t%3);
        if (t == 1) init.start();
        if (t == 2) iters.start();
        cuda_set <<<blocks, threads>>> (dev_base, arr_at(dev_to_send, t, buf_size*6), dev_to_recv, dev_data, t, is_beginning, is_end);

        cudaDeviceSynchronize();
        for (int i = 0; i< 6; i++)
            with_check(cudaMemcpy(to_send[i + sh], arr_at(dev_to_send, t, buf_size*6)+i*buf_size, buf_size*sizeof(double), cudaMemcpyDeviceToHost), "to_send");
        cudaDeviceSynchronize();
        init.stop();
        mpi.start();
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
        mpi.stop();
        gpu.start();
        for (int i = 0; i < 6; i++)
            with_check(cudaMemcpy(arr_at(dev_to_recv, t, buf_size*6)+i*buf_size, to_recv[i + sh], buf_size*sizeof(double), cudaMemcpyHostToDevice), "to_recv");
        gpu.stop();
        if (t > 2) {
            cudaDeviceSynchronize();
            cuda_diff <<<blocks, threads>>> (arr_at(dev_base, t, base_size), dev_data, t, (double*)dev_results+t);
            double my_max = 0;
            cudaDeviceSynchronize();
            gpu.start();
            with_check(cudaMemcpy(&my_max, (double*)dev_results+t, sizeof(double), cudaMemcpyDeviceToHost), "my max"); 
            gpu.stop();
            cout << "my max - " << my_max << endl;
            MPI_Reduce(&my_max, (double*)results+t, 1, MPI_DOUBLE, MPI_MAX, 0, COMM_CART);
        }
        cudaDeviceSynchronize();
    }
    iters.stop();
    finish.start();
    cudaDeviceReset();
    for (int i = 0; i < 3; i++) 
        free(base[i]);
    for (int i = 0; i < 18; i++) {
        free(to_send[i]);
        free(to_recv[i]);
    }
    free(base);
    free(to_send);
    free(to_recv);
    free(data.N);
    free(data.h);
    free(data.base_vals);
    finish.stop();
    common.stop();
    if (myid == 0) {
        cout.precision(4);
        cout << "Work took " << common.get_time() << " seconds." << endl;
        cout << "Initialisation took " << init.get_time() << " seconds." << endl;
        cout << "Iterations took " << iters.get_time() << " seconds." << endl;
        cout << "GPU data exchanges took " << gpu.get_time() << " seconds." << endl;
        cout << "MPI data exchanges took " << mpi.get_time() << " seconds." << endl;
        cout << "Finishing took " << finish.get_time() << " seconds." << endl;
        cout << "Max error = " << scientific << *max_element(results+2, results+(int)ITER) << endl;
    }
    free(results);
    MPI_Finalize();
    return 0;
}
