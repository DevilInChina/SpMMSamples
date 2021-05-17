#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

void swap(int *a, int *b) {
    if (*a != *b)
        *a ^= *b ^= *a ^= *b;
}

int cmp(const void *a, const void *b) {
    return *((int *) a) - *((int *) b);
}

#define NNZ 50
#define WARMUP_TIMES 5
#define VALUE_TYPE double
#define BENCH_TIMES 10

void GenerateCsr(int **RowPtr, int **ColIdx, int m) {
    srand(m);
    *RowPtr = (int *) malloc(sizeof(int) * (m + 1));
    *ColIdx = (int *) malloc(sizeof(int) * (m * NNZ));

    int *randCol = (int *) malloc(sizeof(int) * m * 2);
    for (int i = 0; i < m; ++i) {
        randCol[i] = i;
    }

    for (int i = 0; i < m; ++i) {
        swap(randCol + i, randCol + rand() % m);
    }
    memcpy(randCol + m, randCol, sizeof(int) * m);
    (*RowPtr)[0] = 0;
    for (int i = 1; i <= m; ++i) {
        int nnz = rand() % NNZ + 1;
        if (nnz > m)nnz = m;
        (*RowPtr)[i] = (*RowPtr)[i - 1] + nnz;
        int buff = rand() % m;
        memcpy(*ColIdx + (*RowPtr)[i - 1], randCol + buff, nnz * sizeof(int));
        qsort(*ColIdx + (*RowPtr)[i - 1], nnz, sizeof(int), cmp);
    }

    *ColIdx = (int *) realloc(*ColIdx, sizeof(int) * (*RowPtr)[m]);
    free(randCol);
}

void GeMM(int m, int width,
          VALUE_TYPE *MatrixVal, VALUE_TYPE *denseRightMatrix,
          VALUE_TYPE *Res, double *time_val) {

    struct timeval t1, t2;
    *time_val = 0;
    for (int _ = 0; _ < BENCH_TIMES; ++_) {
        memset(Res, 0, sizeof(VALUE_TYPE) * width * m);
        gettimeofday(&t1, NULL);
#pragma omp parallel for
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                for (int k = 0; k < width; ++k) {
                    Res[i * width + k] += MatrixVal[i * m + j] * denseRightMatrix[j * width + k];
                }
            }
        }
        gettimeofday(&t2, NULL);
        *time_val += ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0);
    }
    *time_val /= BENCH_TIMES;
}

void csrSpMM(int m, int *RowPtr, int *ColIdx, VALUE_TYPE *CsrVal,
             int width, VALUE_TYPE *denseRightMatrix, VALUE_TYPE *Res, double *time_val) {

    struct timeval t1, t2;
    *time_val = 0;
    for (int _ = 0; _ < BENCH_TIMES; ++_) {
        memset(Res, 0, sizeof(VALUE_TYPE) * width * m);
        gettimeofday(&t1, NULL);
#pragma omp parallel for
        for (int i = 0; i < m; ++i) {
            for (int j = RowPtr[i]; j < RowPtr[i + 1]; ++j) {
                for (int k = 0; k < width; ++k) {
                    Res[i * width + k] += CsrVal[j] * denseRightMatrix[ColIdx[j] * width + k];
                }
            }
        }
        gettimeofday(&t2, NULL);
        *time_val += ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0);
    }
    *time_val /= BENCH_TIMES;
}

void compareUndPrint(const char *name, const double *C_Golden, const double *C_ref, int m, int n) {

    int count1 = 0;
    for (int i = 0; i < m * n; i++)
        if (C_Golden[i] != C_ref[i]) {
            //    printf("%d %d %f %f\n",i/n,i%n,C[i],C_golden[i]);
            count1++;
        }
    if (count1 == 0)
        printf("(%s)(row-col, A and B are in row-major) PASS!\n\n", name);
    else
        printf("(%s)(row-col, A and B are in row-major) NOT PASS!\n\n", name);
}

__global__ void SpMMKernel(int m, int *RowPtr, int *ColIdx, VALUE_TYPE *CsrVal,
                           int width, VALUE_TYPE *denseRightMatrix, VALUE_TYPE *Res) {
// Each thread computes one element of C
// by accumulating results into Cvalue
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    //int col = blockIdx.y * blockDim.y + threadIdx.y;
    for(int k = 0 ; k < width ; ++k) {
        Res[row * width + k] = 0;
    }
    for (int j = RowPtr[row]; j < RowPtr[row + 1]; ++j) {
        double val = 0;
        for(int k = 0 ; k < width ; ++k) {
            Res[row * width + k] += CsrVal[j] * denseRightMatrix[ColIdx[j] * width + k];
        }
    }
}


void spMM_cuda_yours(int m, int *RowPtr, int *ColIdx, VALUE_TYPE *CsrVal,
                     int width, VALUE_TYPE *denseRightMatrix, VALUE_TYPE *Res, double *time_value) {

    int *d_RowPtr, *d_ColIdx;

    size_t size = (m + 1) * sizeof(int);
    cudaMalloc(&d_RowPtr, size);
    cudaMemcpy(d_RowPtr, RowPtr, size,
               cudaMemcpyHostToDevice);

    size = RowPtr[m] * sizeof(int);
    cudaMalloc(&d_ColIdx, size);
    cudaMemcpy(d_ColIdx, ColIdx, size,
               cudaMemcpyHostToDevice);
// Allocate C in device memory

    double *d_CsrVal, *d_denseRightMatrix, *d_Res;
    size = RowPtr[m] * sizeof(double);
    cudaMalloc(&d_CsrVal, size);
    cudaMemcpy(d_CsrVal, CsrVal, size,
               cudaMemcpyHostToDevice);

    size = sizeof(double) * m * width;

    cudaMalloc(&d_denseRightMatrix, size);
    cudaMemcpy(d_denseRightMatrix, denseRightMatrix, size,
               cudaMemcpyHostToDevice);

    cudaMalloc(&d_Res, size);
    dim3 dimBlock(1, 1);
    dim3 dimGrid(m, 1);

    for (int i = 0; i < WARMUP_TIMES; ++i) {

        ///// edit your warmup code here

        SpMMKernel<<<dimGrid,dimBlock>>>(m,d_RowPtr,d_ColIdx,d_CsrVal,
                                         width,d_denseRightMatrix,d_Res);
        ////
    }
    timeval t1, t2;
    cudaDeviceSynchronize();
    *time_value = 0;
    for (int i = 0; i < BENCH_TIMES; ++i) {
        // cublasSgemm('N', 'N', m, n, k, 1.0f, d_A, m, d_B, k, 0, d_C, m);

        gettimeofday(&t1, nullptr);
        ///// edit your code here

        SpMMKernel<<<dimGrid,dimBlock>>>(m,d_RowPtr,d_ColIdx,d_CsrVal,
                                         width,d_denseRightMatrix,d_Res);


        ////

        cudaDeviceSynchronize();
        gettimeofday(&t2, nullptr);
        *time_value += (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0;
    }


    *time_value /= BENCH_TIMES;
    cudaMemcpy(Res, d_Res, size,
               cudaMemcpyDeviceToHost);

    cudaFree(d_ColIdx);
    cudaFree(d_Res);
    cudaFree(d_CsrVal);
    cudaFree(d_RowPtr);
    cudaFree(d_denseRightMatrix);

}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("First parameter is height and width of left matrix.\n"
               "Second parameter is width of right matrix(8,16,32 is recommended).");
        exit(0);
    }
    int m = atoi(argv[1]);
    int width = atoi(argv[2]);
    int *RowPtr, *ColIdx;
    GenerateCsr(&RowPtr, &ColIdx, m);
    VALUE_TYPE *CsrVal = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * (RowPtr[m]));
    for (int i = 0; i < RowPtr[m]; ++i) {
        CsrVal[i] = (rand() % 8 + 1) / 8.0;
    }

    VALUE_TYPE *DenseMatrixVal = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * m * m);
    memset(DenseMatrixVal, 0, sizeof(VALUE_TYPE) * m * m);

    for (int i = 0; i < m; ++i) {
        for (int j = RowPtr[i]; j < RowPtr[i + 1]; ++j) {
            DenseMatrixVal[i * m + ColIdx[j]] = CsrVal[j];
        }
    }

    VALUE_TYPE *RightThinMatrix = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * width * m);
    srand(width);
    for (int i = 0; i < width * m; ++i) {
        RightThinMatrix[i] = rand() % 32 * 0.125;
    }
    VALUE_TYPE *Res_Golden = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * width * m);
    VALUE_TYPE *Res = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * width * m);
    double time_value;

    printf("Matrix A is %i x %i, matrix B is %i x %i\n", m, m, m, width);
    printf("Matrix A has a sparsity of %.3f%%\n", RowPtr[m] * 100.0 / m / m);

    GeMM(m, width, DenseMatrixVal, RightThinMatrix, Res_Golden, &time_value);
    const char *Name = "GeMM";
    printf("\n(%s)(row-col, A and B are in row-major)) used %4.5f ms\n",
           Name, time_value);


    csrSpMM(m, RowPtr, ColIdx, CsrVal, width, RightThinMatrix, Res, &time_value);
    Name = "csrSpMM";
    printf("\n(%s)(row-col, A and B are in row-major)) used %4.5f ms\n",
           Name, time_value);
    compareUndPrint(Name, Res, Res_Golden, m, width);


    spMM_cuda_yours(m, RowPtr, ColIdx, CsrVal, width, RightThinMatrix, Res, &time_value);
    Name = "cudaSpMM";
    printf("\n(%s)(row-col, A and B are in row-major)) used %4.5f ms\n",
           Name, time_value);
    compareUndPrint(Name, Res, Res_Golden, m, width);

    return 0;
}
