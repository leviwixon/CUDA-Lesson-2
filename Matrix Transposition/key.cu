
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

const int BLOCK_DIM = 4;   // used to set dimensions of a block (threads per block).

// Function meant for outputting the stats & info of the matricies. Inclusion of '__host__' is optional, but helps create visual
// seperation between '__global__', '__device__', '__host__ __device', etc. For more info on execution space specifiers and how
// they work, check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-declaration-specifiers
__host__ 
void formattedPrint(int* matrix, long double totalTime, long double averageTime, int iterations, int rows, int cols) {
    printf("Total time for %d iterations: %f (ms)\n", iterations, totalTime);
    printf("Avg time per iteration: %f (ms)\n", averageTime);
    for (unsigned int i = 0; i < rows; i++) {
        for (unsigned int j = 0; j < cols; j++) {
            printf("%3d ", matrix[(i * cols + j)]);
        }
        printf("\n");
    }
}

// Used to test whether or not the matrix transposition worked.
__host__ 
bool testTransposition(int* originalMatrix, int* TMatrix, unsigned int originalRows, unsigned int originalCols) {
    for (unsigned int i = 0; i < originalRows; i++) {
        for (unsigned int j = 0; j < originalCols; j++) {
            if (originalMatrix[i * originalCols + j] != TMatrix[j * originalRows + i]) {
                //printf("Failed on index %d - %d and %d - %d\n", i * originalCols, j, j * originalRows, i);
                return false;
            }
        }
    }
    return true;
}

/* This function was taken from https ://github.com/JonathanWatkins/CUDA/blob/master/NvidiaCourse/Exercises/transpose/transpose.cu
** and provides a significantly faster overall transposition time when dealing with larger array sizes. The accompanying comments
** provided for the transpose function in Jonathan Watkins' transpose.cu file provide some elaboration as to why this is done, and
** where the benefit comes from. For a simpler, albeit slower implementation of matrix transposition, refer to the 'transpose_naive'
** function within the github link, or the countless other naive tranpositions available on the internet. You could design a function
** which is completed with less threads by picking up where the last thread left off, however CUDA provides such a stupidly high upper
** limit, that it really isn't necessary on the hardware and situation we have. It is theoretically possible to have a grid of blocks
** with dimension 65,535 * 65,535 * 65,535 with 1,024 per thread, or about 288 quadrillion threads. Obviously, most don't have the req
** hardware for that, but it gives a good idea of just how much we can throw at a given task. Sources for the math on upper limit in CUDA
** can be found at https://stackoverflow.com/questions/12078080/max-number-of-threads-which-can-be-initiated-in-a-single-cuda-kernel, but
** is also subject to the GPU's compute capability, as seen in https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
*/
__global__ 
void gpu_parallel_transpose_blocks(int* odata, int* idata, int width, int height) {
    __shared__ int block[BLOCK_DIM][BLOCK_DIM + 1];
    // read the matrix tile into shared memory
    // load one element per thread from device memory (idata) and store it
    // in transposed order in block[][]
    unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
    unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
    if ((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[threadIdx.y][threadIdx.x] = idata[index_in];
    }

    // synchronise to ensure all writes to block[][] have completed
    __syncthreads();    // False error by intellisense is possible here. If you see "identifier '__syncthreads()' is undefined", it likely will still run. 

    // write the transposed matrix tile to global memory (odata) in linear order
    xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
    if ((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = block[threadIdx.x][threadIdx.y];
    }
}

int main()
{
    srand(4);   // static seed for testing!

    // Timing variables
    cudaEvent_t start, stop;
    float elapsedTime, avgTime;
    int totalIterations = 10000;

    int dimFlag = 0;
    unsigned int m, n;   
     do {
        printf("Please enter row dimension for first matrix: ");
        if (scanf("%d", &m) != 1) {
            printf("ERROR: Failed to read integer!\n");
            return -1; 
        }
        
        printf("Please enter column dimension for first matrix: ");
        if (scanf("%d", &n) != 1) {
            printf("ERROR: Failed to read integer!\n");
            return -1;
        }

        if (!m || !n) {
            printf("ERROR: Cannot have dimension of value 0! Please try again with a valid dimension (>0).\n");
            m = n = 0;  // set them all to 0, and try again.
            dimFlag = 1;
        }

        else {
            dimFlag = 0;
        }
     } while (dimFlag);

    const int mem_size = m * n * sizeof(int);

    // Make Host Side Matricies
    int* matrixA, * matrixB;
    cudaMallocHost((void**)&matrixA, mem_size);
    cudaMallocHost((void**)&matrixB, mem_size);

    // Make Device Side Matricies
    int* devMatrixA, * devMatrixB;
    cudaMalloc((void**)&devMatrixA, mem_size);
    cudaMalloc((void**)&devMatrixB, mem_size);

    // Instantiate Host Side Matrix
    for (unsigned int i = 0; i < m; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            matrixA[i * n + j] = rand() % 100;
        }
    }

    // Optional output for comparison to transposed matrix later on
    /*for (unsigned int i = 0; i < m; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            printf("%3d ", matrixA[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");*/

    // Copy Host Side Matricies to Device Side
    cudaMemcpy(devMatrixA, matrixA, mem_size, cudaMemcpyHostToDevice);

    // Set dimensions for the device work done later
    dim3 threads(BLOCK_DIM, BLOCK_DIM);   
    dim3 grid(m/BLOCK_DIM, n/BLOCK_DIM, 1); 

    // Cuda timing via events documented at https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#using-cuda-gpu-timers
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // Run many iterations to make tangible times more likely in small matrix cases (small being relative to the amount of possible threads).
    for (int i = 0; i < totalIterations; i++) {
        gpu_parallel_transpose_blocks <<<grid, threads >>> (devMatrixB, devMatrixA, n, m); // ignore "expected an expression" error if it appears. It is a false flag by intellisense.
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    avgTime = elapsedTime / totalIterations;

    // Copy result in from Device and print out statistics for the run.
    cudaMemcpy(matrixB, devMatrixB, mem_size, cudaMemcpyDeviceToHost);
    if (testTransposition(matrixA, matrixB, m, n)) {
        //formattedPrint(matrixB, elapsedTime, avgTime, totalIterations, n, m);
    }
    else {
        printf("ERROR: Matrix transposition failed!");
    }

    /*ANY ADDITIONAL MEMORY MUST BE FREED BEFORE LEAVING*/
    cudaFree(devMatrixA);
    cudaFree(devMatrixB);
    cudaFreeHost(matrixA);
    cudaFreeHost(matrixB);
    /****************************************************/
}
