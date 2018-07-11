#ifndef CUML_LEARN_LA_H
#define CUML_LEARN_LA_H

#include <assert.h>
#include "../utils.h"
#include <iostream>
#include "cublas_v2.h"


extern cublasHandle_t cublas_handle;

__global__
void printVec(int N, float* data){
	gout<<gendl;
	for (int i = 0;i<N;++i){
		gout<<*(data+i)<<gendl;
	}
	gout<<gendl;
}

__global__
void printMat(int rows,int cols, float* data){
	gout<<gendl;
	for (int r=0;r<rows;++r){
		for(int c = 0;c<cols;++c){
			gout<<*(data+r+c*rows)<<"  ";
		}
		gout<<gendl;
	}
	gout<<gendl;
}

struct VectorF{
	int N;
	float *data = nullptr;

	VectorF(int N_):N(N_){
		initData();
	}

	void copyFrom(int N_, float* data_){
		N = N_;
		HANDLE_ERROR( cudaMemcpy(data,data_,N*sizeof(*data), cudaMemcpyDeviceToDevice) );
	}

	VectorF(int N_, float* data_){
		N = N_;
		initData();
		copyFrom(N_,data_);
	}

	//copy constructor
	VectorF(const VectorF& v){
		N = v.N;
		initData();
		copyFrom(v.N,v.data);
	}

	//copy assignment
	VectorF& operator=(const VectorF& v){
		copyFrom(v.N,v.data);
		return *this;
	}

	//move constructor
	VectorF(VectorF&& v):N(v.N),data(v.data){
		v.data = nullptr;
	}

	//move assignment
	VectorF& operator=(VectorF&& v){
		N = v.N;
		data = v.data;
		v.data = nullptr;
		return *this;
	}

	void initData(){
		HANDLE_ERROR( cudaMalloc (&data,N*sizeof(*data)) );
	}

	void free(){
		HANDLE_ERROR( cudaFree(data) );
	}

	void print(){
		printVec<<<1,1>>>(N,data);
	}

	~VectorF(){
		free();
	}

	float length(){
		float result;
		HANDLE_ERROR( cublasSnrm2(cublas_handle,N,data,1,&result) );
		return result;
	}

};


static VectorF newVectorFromRAM(int N, float* data){
	VectorF v(N);
	HANDLE_ERROR( cudaMemcpy(v.data,data,N*sizeof(*data), cudaMemcpyHostToDevice) );
	return v;
}


struct MatrixF{
	int rows;
	int cols;
	float* data;

	MatrixF(int rows_,int cols_):rows(rows_),cols(cols_){
		initData();
	}

	void copyFrom(int rows_,int cols_, float* data_){
		rows = rows_;
		cols = cols_;
		HANDLE_ERROR( cudaMemcpy(data,data_,rows*cols*sizeof(*data), cudaMemcpyDeviceToDevice) );
	}

	MatrixF(int rows_,int cols_, float* data_){
		rows = rows_;
		cols = cols_;
		initData();
		copyFrom(rows_,cols_,data_);
	}

	//copy constructor
	MatrixF(const MatrixF& m){
		rows = m.rows;
		cols = m.cols;
		initData();
		copyFrom(m.rows,m.cols,m.data);
	}

	//copy assignment
	MatrixF& operator=(const MatrixF& m){
		copyFrom(m.rows,m.cols,m.data);
		return *this;
	}

	//move constructor
	MatrixF(MatrixF&& m):rows(m.rows),cols(m.cols),data(m.data){
		m.data = nullptr;
	}

	//move assignment
	MatrixF& operator=(MatrixF&& m){
		rows = m.rows;
		cols = m.cols;
		data = m.data;
		m.data = nullptr;
		return *this;
	}


	void initData(){
		HANDLE_ERROR( cudaMalloc (&data,rows*cols*sizeof(*data)) );
	}

	void free(){
		HANDLE_ERROR( cudaFree(data) );
	}

	void print(){
		printMat<<<1,1>>>(rows,cols,data);
	}

	~MatrixF(){
		free();
	}

	MatrixF transpose(){
		MatrixF result(cols,rows);
		float one = 1.f;
		float zero = 0.f;
		HANDLE_ERROR( cublasSgeam(cublas_handle,CUBLAS_OP_T,CUBLAS_OP_T,cols,rows,&one,data,rows,&zero,data,rows,result.data,result.rows) );
		return result;
	}
};

static MatrixF newMatrixFromRAM(int rows,int cols, float* data){
	MatrixF m(rows,cols);
	cudaMemcpy(m.data,data,rows*cols*sizeof(*data), cudaMemcpyHostToDevice);
	return m;
}



static float dot(VectorF& a,VectorF& b){
	assert (a.N==b.N);
	float result;
	HANDLE_ERROR (cublasSdot(cublas_handle,a.N,a.data,1,b.data,1,&result) );
	return result;
}


static VectorF operator+ (const VectorF& v1,const VectorF& v2){
	assert (v2.N==v1.N);
	VectorF result(v1);
	float one = 1.f;
	HANDLE_ERROR ( cublasSaxpy(cublas_handle,v1.N,&one,v2.data,1,result.data,1)  );
	return result;
}

static VectorF operator- (const VectorF& v1,const VectorF& v2){
	assert (v2.N==v1.N);
	VectorF result(v1);
	float neg_one = -1.f;
	HANDLE_ERROR ( cublasSaxpy(cublas_handle,v1.N,&neg_one,v2.data,1,result.data,1)  );
	return result;
}

static VectorF operator-(const VectorF& v){
	VectorF result(v);
	float r = -1;
	HANDLE_ERROR ( cublasSscal(cublas_handle,v.N,&r,result.data,1)  );
	return result;
}



static VectorF operator*(const VectorF& v, float f){
	VectorF result(v);
	float r = f;
	HANDLE_ERROR ( cublasSscal(cublas_handle,v.N,&r,result.data,1)  );
	return result;
}

static VectorF operator*( float f, const VectorF& v){
	return v*f;
}

static MatrixF operator+ (const MatrixF& m1,const MatrixF& m2){
	assert (m2.rows==m1.rows);
	assert (m2.cols==m1.cols);
	MatrixF result(m1);
	float one = 1.f;
	HANDLE_ERROR ( cublasSaxpy(cublas_handle,m1.rows*m1.cols,&one,m2.data,1,result.data,1)  );
	return result;
}

static MatrixF operator- (const MatrixF& m1,const MatrixF& m2){
	assert (m2.rows==m1.rows);
	assert (m2.cols==m1.cols);
	MatrixF result(m1);
	float neg_one = -1.f;
	HANDLE_ERROR ( cublasSaxpy(cublas_handle,m1.rows*m1.cols,&neg_one,m2.data,1,result.data,1)  );
	return result;
}

static MatrixF operator-(const MatrixF& m){
	MatrixF result(m);
	float r = -1;
	HANDLE_ERROR ( cublasSscal(cublas_handle,m.rows*m.cols,&r,result.data,1)   );
	return result;
}


static MatrixF operator*(const MatrixF& m, float f){
	MatrixF result(m);
	float r = f;
	HANDLE_ERROR ( cublasSscal(cublas_handle,m.rows*m.cols,&r,result.data,1)   );
	return result;
}

static MatrixF operator*( float f, const MatrixF& m){
	return m*f;
}

static MatrixF operator* (const MatrixF& m1,const MatrixF& m2){
	assert (m1.cols==m2.rows);
	MatrixF result(m1.rows,m2.cols);
	float one = 1.f;
	float zero = 0.f;
	HANDLE_ERROR ( cublasSgemm(cublas_handle,CUBLAS_OP_N,CUBLAS_OP_N,m1.rows,m2.cols,m1.cols,&one,m1.data,m1.rows,m2.data,m2.rows,&zero,result.data,result.rows)  );
	return result;
}

static VectorF operator* (const MatrixF& m,const VectorF& v){
	assert (m.cols==v.N);
	VectorF result(m.rows);
	float one = 1.f;
	float zero = 0.f;
	HANDLE_ERROR ( cublasSgemv(cublas_handle,CUBLAS_OP_N,m.rows,m.cols,&one,m.data,m.rows,v.data,1,&zero, result.data,1)  );
	return result;
}


#endif
