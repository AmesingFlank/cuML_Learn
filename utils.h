//
// Created by AmesingFlank on 2018/7/8.
//

#ifndef CUML_LEARN_UTILS_H
#define CUML_LEARN_UTILS_H

#include <stdio.h>
#include <cstdlib>
#include "cublas_v2.h"


const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}


static void HandleError( cudaError_t err,const char *file,int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

static void HandleError( cublasStatus_t err,const char *file,int line ) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        printf( "%s in %s at line %d\n", cublasGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define HANDLE_NULL( a ) {if (a == NULL) { \ printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

struct endl_flag{};
struct ostream_gpu{
	__device__ ostream_gpu& operator<<(const char* text){
		printf(text);
		return *this;
	};
	__device__ ostream_gpu& operator<<(int text){
		printf("%d",text);
		return *this;
	};
	__device__ ostream_gpu& operator<<(float text){
		printf("%.2f",text);
		return *this;
	};
	__device__ ostream_gpu& operator<<(endl_flag _){
		printf("\n");
		return *this;
	};
};

extern cublasHandle_t cublas_handle;

static void initCUDA(){
	cublasCreate (& cublas_handle );
}

__constant__ ostream_gpu gout;
__constant__ endl_flag gendl;

#endif //CUML_LEARN_UTILS_H
