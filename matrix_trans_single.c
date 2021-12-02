#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int matrix [32001][32001];
int transposedMatrix [32001][32001];

//function fills the matrix with the correct amount of random integers
int fillMatrix(int rows, int seed)
{
    int i, j;

    srand(seed);

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < rows; j++)
        {
            matrix[i][j] = rand() % 100;
        }
    }

    return 0;
}

//function prints the original matrix for testing purposes
int printMatrix(int rows)
{
    int i, j;

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < rows; j++)
        {
            printf("%4d", matrix[i][j]);
        }
        printf("\n\n");
    }

    return 0;
}

//function prints the transposed matrix for testing purposes
int printTransMatrix(int rows)
{
    int i, j;

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < rows; j++)
        {
            printf("%4d", transposedMatrix[i][j]);
        }
        printf("\n\n");
    }

    return 0;
}

//function creates a transposed matrix from the original matrix
int transposeMatrix(int rows)
{
    int i, j;

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < rows; j++)
        {
            transposedMatrix[i][j] = matrix[j][i];
        }
    }
}

int main()
{
    int rows;
    int goodRows;
    int seed;
    int valid;
    clock_t startTime;
    clock_t endTime;
    double totalTime;

    valid = 0;
    goodRows = 0;

    //asks user for number of rows and a seed to use
    printf("Welcome to Matrix Transposition Single Thread!\n");
    printf("----------------------------------------------\n");
    while (goodRows == 0)
    {
        printf("Please enter the number of rows for the square matrix. (1-32000)\n");
        scanf("%d", &rows);
        if (rows > 0 && rows < 32001)
        {
            goodRows = 1;
        }
    }
    printf("Please enter a random seed to test.\n");
    scanf("%d", &seed);

    //commented out lines are for printing matrices for testing
    
    // printf("\n");
    // printf("Starting Matrix\n\n");

    fillMatrix(rows, seed);
    // printMatrix(rows);

    // printf("\n");
    // printf("----------------------------------------------\n");
    // printf("\n");
    // printf("Transposed Matrix\n\n");

    startTime = clock();

    transposeMatrix(rows);
    // printTransMatrix(rows);

    endTime = clock();

    totalTime = (double)(endTime - startTime) / CLOCKS_PER_SEC;

    printf("Matrix Transposed in %.3f seconds\n", totalTime);

    return 0;
}
