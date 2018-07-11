#ifndef CUML_LEARN_LA_H
#define CUML_LEARN_LA_H

#include <assert.h>
#include "../utils.h"
#include <iostream>
#include "cublas_v2.h"


extern cublasHandle_t cublas_handle;
extern int dataCount;

void printDataCount(){

}

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


template <typename Func>
__global__
void unaryExprKernel(float* data,Func f){
	int offset = threadIdx.x;
	data[offset] = f(data[offset]);
}

template <typename Func>
__global__
void binaryOperatorKernel(float* data1,float* data2, float* data3, Func f){
	int offset = threadIdx.x;
	data3[offset] = f(data1[offset],data2[offset]);
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
		free();
		N = v.N;
		initData();
		copyFrom(v.N,v.data);
		return *this;
	}

	//move constructor
	VectorF(VectorF&& v):N(v.N),data(v.data){
		v.data = nullptr;
	}

	//move assignment
	VectorF& operator=(VectorF&& v){
		free();
		N = v.N;
		data = v.data;
		v.data = nullptr;
		return *this;
	}

	void initData(){
		HANDLE_ERROR( cudaMalloc (&data,N*sizeof(*data)) );
		++dataCount;
		printDataCount();
	}

	void free(){
		HANDLE_ERROR( cudaFree(data) );
		if (data!=nullptr) --dataCount;
		printDataCount();
		data = nullptr;
	}

	void print(){
		printVec<<<1,1>>>(N,data);
	}

	~VectorF(){
		free();
	}

	float modulus(){
		float result;
		HANDLE_ERROR( cublasSnrm2(cublas_handle,N,data,1,&result) );
		return result;
	}

	float operator[](int i){
		float result = 0;
		HANDLE_ERROR( cudaMemcpy(&result,data+i,sizeof(*data), cudaMemcpyDeviceToHost) );
		return result;
	}

	void set(int i,float f){
		assert(i<N);
		HANDLE_ERROR( cudaMemcpy(data+i,&f,sizeof(*data), cudaMemcpyHostToDevice) );
	}

	template <typename Func>
	VectorF unaryExpr(Func f){
		VectorF v(N,data);
		unaryExprKernel<<<1,v.N>>>(v.data,f);
		return v;
	}

};


static VectorF newVectorFromRAM(int N, float* data){
	VectorF v(N);
	HANDLE_ERROR( cudaMemcpy(v.data,data,N*sizeof(*data), cudaMemcpyHostToDevice) );
	return v;
}
static VectorF newZeroVector(int N){
	VectorF v(N);
	HANDLE_ERROR( cudaMemset(v.data,0,N*sizeof(*v.data)) );
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
		free();
		rows = m.rows;
		cols = m.cols;
		initData();
		copyFrom(m.rows,m.cols,m.data);
		return *this;
	}

	//move constructor
	MatrixF(MatrixF&& m):rows(m.rows),cols(m.cols),data(m.data){
		m.data = nullptr;
	}

	//move assignment
	MatrixF& operator=(MatrixF&& m){
		free();
		rows = m.rows;
		cols = m.cols;
		data = m.data;
		m.data = nullptr;
		return *this;
	}


	void initData(){
		HANDLE_ERROR( cudaMalloc (&data,rows*cols*sizeof(*data)) );
		++dataCount;
		printDataCount();
	}

	void free(){
		HANDLE_ERROR( cudaFree(data) );
		if (data!=nullptr) --dataCount;
		printDataCount();
		data = nullptr;

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

	template <typename Func>
	MatrixF unaryExpr(Func f){
		MatrixF v(rows,cols,data);
		unaryExprKernel<<<1,rows*cols>>>(v.data,f);
		return v;
	}
};

static MatrixF newMatrixFromRAM(int rows,int cols, float* data){
	MatrixF m(rows,cols);
	cudaMemcpy(m.data,data,rows*cols*sizeof(*data), cudaMemcpyHostToDevice);
	return m;
}

static MatrixF newZeroMatrix(int rows,int cols){
	MatrixF m(rows,cols);
	HANDLE_ERROR( cudaMemset(m.data,0,rows*cols*sizeof(*m.data)) );
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

static VectorF operator/(const VectorF& v, float f){
	VectorF result(v);
	float r = 1.f/f;
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

static MatrixF operator/(const MatrixF& m, float f){
	MatrixF result(m);
	float r = 1.f/f;
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

template<typename Func>
static VectorF binaryOperator(const VectorF& v1,const VectorF& v2, Func f){
	assert(v1.N==v2.N);
	VectorF result(v1.N);
	binaryOperatorKernel<<<1,v1.N>>>(v1.data,v2.data,result.data,f);
	return result;
}

template<typename Func>
static MatrixF binaryOperator(const MatrixF& m1,const MatrixF& m2, Func f){
	assert(m1.rows==m2.rows);
	assert(m1.cols==m2.cols);
	MatrixF result(m1.rows,m1.cols);
	binaryOperatorKernel<<<1,m1.rows*m1.cols>>>(m1.data,m2.data,result.data,f);
	return result;
}

MatrixF columnMatrix(VectorF vec){
	MatrixF result(vec.N,1);
	result.copyFrom(vec.N,1,vec.data);
	return result;
}
MatrixF rowMatrix(VectorF vec){
	return columnMatrix(vec).transpose();
}

VectorF cwiseProduct(VectorF v1,VectorF v2){
	auto lambda = []__device__(float f1,float f2)->float{
		return f1*f2;
	};
	return binaryOperator(v1,v2,lambda);
}


#endif
