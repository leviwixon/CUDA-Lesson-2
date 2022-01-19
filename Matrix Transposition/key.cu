#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>


const int BLOCK_DIM = 16;   // used to set dimensions of a block (threads per block). Typically best as 2^N

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
                return false;
            }
        }
    }
    return true;
}

__global__
void naive_parallel_transposition(int* matrix, int* Tmatrix, int rows, int cols) {
    // essential becomes "(width of array) * (array number) + (array index)". This allows roll-over (e.g. an array of 8 threads with 5
    // blocks will reach a 9th element and jump to the new array, but record that the width of the previous has been traversed, or more
    // succintly, we have traverse (width * (total num of array traversed)) elements/threads).
    int xCoord = blockDim.x * blockIdx.x + threadIdx.x;
    int yCoord = blockDim.y * blockIdx.y + threadIdx.y;
    if (xCoord < cols && yCoord < rows) {       // Its possible to end up with some straggler threads that would go beyond the bounds needed, so we cull them with this
        // 1D writing of typicaly 2D array access (think of it as row == column_size * row_index (e.g coordinates in an 8x8 array [6][7] == [(6 * 8) + 7] == [48 + 7] == [55])
        Tmatrix[yCoord + rows * xCoord] = matrix[xCoord + cols * yCoord];  
    }
}

int main()
{
    srand(4);   // static seed for testing!
    // Timing variables
    cudaEvent_t start, stop;
    float elapsedTime, avgTime;
    int totalIterations = 1;

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
    double xDim, yDim;

    /* UNCOMMET FOR DYNAMIC BLOCK ASSIGNMENT*/
    //dim3 threads(BLOCK_DIM, BLOCK_DIM);   
    //xDim = (m + (BLOCK_DIM - 1)) / BLOCK_DIM;   // Just round up on division.
    //yDim = (n + (BLOCK_DIM - 1)) / BLOCK_DIM;
    //dim3 grid(yDim, xDim, 1);

    /* Setup for GPU that can handle 1024 threads per block*/
    dim3 threads(32, 32);
    dim3 grid(1, 1);    // 1 block
    /* RECOMMENT ABOVE BLOCK IF USING DYNAMIC BLOCK ASSIGNMENT*/

    // Cuda timing via events documented at https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#using-cuda-gpu-timers
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // Run many iterations to make tangible times more likely in small matrix cases (small being relative to the amount of possible threads).
    for (int i = 0; i < totalIterations; i++) {
        naive_parallel_transposition <<<grid, threads >>> (devMatrixA, devMatrixB, m, n); // ignore "expected an expression" error if it appears. It is a false flag by intellisense.
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
        printf("SUCCESS!\n");
    }
    else {
        printf("ERROR: Matrix transposition failed!\n");
    }

    /*ANY ADDITIONAL MEMORY MUST BE FREED BEFORE LEAVING*/
    cudaFree(devMatrixA);
    cudaFree(devMatrixB);
    cudaFreeHost(matrixA);
    cudaFreeHost(matrixB);
    /****************************************************/
}
