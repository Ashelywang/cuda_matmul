#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <complex>
#include <boost/math/constants/constants.hpp>
//#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include "matrix.h"
#include "mex.h"

//using namespace std;
using namespace boost::numeric::ublas;
using namespace boost::numeric::odeint;

const double pi = boost::math::constants::pi<double>();
// --------------------- Newly added ----------------------
#include <cuda_runtime.h>

#include "cublas_v2.h"
#include <curand.h>
#include <curand_kernel.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))


void MatrixProdMatrix(const double* A,const double* B, double * C,int m,int r,int n,double* time_trans,double* time_cal){
    /*
    Input:
        matrix A of size m*r 
        matrix B of size r*m
    Output:
        matrix C = A*B
        time_cal: calculation time
    */
    double* A_gpu;
    double* B_gpu;
    double* C_gpu;

    *time_trans = 0;
    *time_cal = 0;
    const double alf = 1;
    const double bet = 0;
    const double *alpha = &alf;
    const double *beta = &bet;


    clock_t  start = clock();
    cudaMalloc(&A_gpu,m*r*sizeof(double));
    cudaMalloc(&B_gpu,n*r*sizeof(double));
    cudaMalloc(&C_gpu,m*n*sizeof(double));
    cudaMemset(C_gpu,0,m*n*sizeof(double));
    cudaMemcpy(A_gpu,A,r*m*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu,B,r*m*sizeof(double),cudaMemcpyHostToDevice);

    clock_t end = clock();
    *time_trans += (double) (end-start) / CLOCKS_PER_SEC * 1000.0;


    // calculation
    start = clock();
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,r,alpha,A_gpu,m,B_gpu,r,beta,C_gpu,m);
    end = clock();
    *time_cal += (double) (end-start) / CLOCKS_PER_SEC * 1000.0;

    //Transmission
    cudaMemcpy(C,C_gpu,n*m*sizeof(double),cudaMemcpyDeviceToHost);

    cudaFree(C_gpu);
    cudaFree(A_gpu);
    cudaFree(B_gpu);

}

void mexFunction(int nlhs, mxArray *plhs[], int nlrs, const mxArray *prhs[])
{
    mwSize n,r,m;
    m = mxGetM(prhs[0]);
    r = mxGetN(prhs[0]);
    n = mxGetN(prhs[1]);

    double *A;
    double *B;
    // get matrix A and B
    A = (double *)mxGetPr(prhs[0]);
    B = (double *)mxGetPr(prhs[1]);


    double * C;
    
    plhs[0] = mxCreateDoubleMatrix(m, n, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);
    C = mxGetPr(plhs[0]);
    double * time_cal = mxGetPr(plhs[1]);
    double * time_trans = mxGetPr(plhs[2]);

    
    MatrixProdMatrix(A,B,C,m,r,n,time_cal,time_trans);
    cudaDeviceSynchronize();
    return;
}
