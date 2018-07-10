#ifndef CUML_LEARN_LA_H
#define CUML_LEARN_LA_H

#include "../utils.h"
struct VectorF{
	int N;
	float *data;

	__host__
	VectorF(int N_):N(N_){
		HANDLE_ERROR( cudaMalloc (&data,N*sizeof(*data)) );
	}

	__host__
	void copyFrom(int N_, float* data_){
		N = N_;
		free();
		HANDLE_ERROR( cudaMemcpy(data,data_,N*sizeof(*data), cudaMemcpyDeviceToDevice) );
	}

	__host__
	VectorF(int N_, float* data_){
		copyFrom(N_,data_);
	}

	//copy constructor
	__host__
	VectorF(const VectorF& v){
		copyFrom(v.N,v.data);
	}

	//copy assignment
	__host__
	VectorF& operator=(const VectorF& v){
		copyFrom(v.N,v.data);
		return *this;
	}

	//move constructor
	__host__
	VectorF(VectorF&& v):N(v.N),data(v.data){}

	__host__
	void free(){
		HANDLE_ERROR( cudaFree(data) );
	}


};

struct MatrixF{
	int rows;
	int cols;
	float* data;

	__host__
	MatrixF(int rows_,int cols_):rows(rows_),cols(cols_){
		HANDLE_ERROR( cudaMalloc (&data,rows*cols*sizeof(*data)) );
	}

	__host__
	void copyFrom(int rows_,int cols_, float* data_){
		rows = rows_;
		cols = cols_;
		free();
		HANDLE_ERROR( cudaMemcpy(data,data_,rows*cols*sizeof(*data), cudaMemcpyDeviceToDevice) );
	}

	__host__
	MatrixF(int rows_,int cols_, float* data_){
		copyFrom(rows_,cols_,data_);
	}

	//copy constructor
	__host__
	MatrixF(const MatrixF& m){
		copyFrom(m.rows,m.cols,m.data);
	}

	//copy assignment
	__host__
	MatrixF& operator=(const MatrixF& m){
		copyFrom(m.rows,m.cols,m.data);
		return *this;
	}

	//move constructor
	__host__
	MatrixF(MatrixF&& m):rows(m.rows),cols(m.cols),data(m.data){}

	__host__
	void free(){
		HANDLE_ERROR( cudaFree(data) );
	}
};


#endif
