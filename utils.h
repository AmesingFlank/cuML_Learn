//
// Created by AmesingFlank on 2018/7/8.
//

#ifndef CUML_LEARN_UTILS_H
#define CUML_LEARN_UTILS_H

#include <stdio.h>
#include <cstdlib>



static void HandleError( cudaError_t err,const char *file,int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
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
		printf("%d",text);
		return *this;
	};
};


__constant__ ostream_gpu gout;

#endif //CUML_LEARN_UTILS_H
