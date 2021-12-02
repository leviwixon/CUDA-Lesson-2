#include <stdio.h>
#include <time.h>
#include <stdlib.h>
//#include <Windows.h>
#include <unistd.h>

int matrix [10001][10001];
int transposedMatrix [10001][10001];

//function fills the matrix with the correct amount of random integers
void fillMatrix(int rows, int cols, int seed)
{
    int i, j;

    srand(seed);

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            matrix[i][j] = rand() % 100;
        }
    }

    return;
}

//function prints the original matrix for testing purposes
void printMatrix(int rows, int cols)
{
    int i, j;

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            printf("%4d", matrix[i][j]);
        }
        printf("\n\n");
    }

    return;
}

//function prints the transposed matrix for testing purposes
void printTransMatrix(int rows, int cols)
{
    int i, j;

    for (i = 0; i < cols; i++)
    {
        for (j = 0; j < rows; j++)
        {
            printf("%4d", transposedMatrix[i][j]);
        }
        printf("\n\n");
    }

    return;
}

//function creates a transposed matrix from the original matrix, sleep added to slow down the function
void transposeMatrix(int rows, int cols)
{
    int i, j;

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            transposedMatrix[j][i] = matrix[i][j];
            sleep(transposedMatrix[j][i] / 100);
        }
    }
    return;
}

//function checks to see if transpose successfully worked
int transposeCheck(int rows, int cols)
{
    int i, j;

    j = 0;

    for (i = 0; i < rows; i++)
    {
        if (transposedMatrix[j][i] != matrix[i][j])
        {
            return 0;
        }
    }

    return 1;
}


int main()
{
    int rows;
    int cols;
    int goodRows;
    int goodCols;
    int seed;
    int valid;
    clock_t startTime;
    clock_t endTime;
    double totalTime;
    int success;

    valid = 0;
    goodRows = 0;
    goodCols = 0;

    //asks user for number of rows and a seed to use
    printf("Welcome to Matrix Transposition Single Thread!\n");
    printf("----------------------------------------------\n");
    while (goodRows == 0)
    {
        printf("Please enter the number of rows for the matrix. (1-10000)\n");
        scanf("%d", &rows);
        if (rows > 0 && rows < 10001)
        {
            goodRows = 1;
        }
    }
        while (goodCols == 0)
    {
        printf("Please enter the number of columns for the matrix. (1-10000)\n");
        scanf("%d", &cols);
        if (cols > 0 && cols < 10001)
        {
            goodCols = 1;
        }
    }
    printf("Please enter a random seed to test.\n");
    scanf("%d", &seed);

    //commented out lines are for printing matrices for testing

    // printf("\n");
    // printf("Starting Matrix\n\n");

    fillMatrix(rows, cols, seed);
    // printMatrix(rows, cols);

    // printf("\n");
    // printf("----------------------------------------------\n");
    // printf("\n");
    // printf("Transposed Matrix\n\n");

    startTime = clock();

    transposeMatrix(rows, cols);
    // printTransMatrix(rows, cols);

    endTime = clock();

    success = transposeCheck(rows, cols);

    totalTime = (double)(endTime - startTime) / CLOCKS_PER_SEC;

    if (success == 1)
    {
        printf("Matrix Successfully Transposed in %.3f seconds\n", totalTime);
    }
    else
    {
        printf("Matrix Unsuccessfully Transposed in %.3f second\n", totalTime);
    }

    return 0;
}
