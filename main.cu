/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */


#include "utils.h"
#include <vector>
#include <iostream>
#include "NN/LA.h"
using namespace std;

cublasHandle_t cublas_handle = nullptr;

int main( void ) {
	initCUDA();

	std::cout<<"Hello Cuda"<<std::endl;

	float f1[3] = {1.f,2.f,3.f};
	VectorF v1 = newVectorFromRAM(3,f1);
	v1.print();

	float f2[3] = {4.f,5.f,6.f};
	VectorF v2 = newVectorFromRAM(3,f2);
	v2.print();

	cout<<dot(v1,v2)<<endl;

	VectorF v3 = v1+v2;
	v3.print();

	VectorF v4 = 3*v1;
	v4.print();

	float mf1[9] = {1,2,3,4,5,6,7,8,9};
	MatrixF m1 = newMatrixFromRAM(3,3,mf1);
	m1.print();

	float mf2[9] = {1,2,3,4,5,6,7,8,9};
	MatrixF m2 = newMatrixFromRAM(3,3,mf2);
	m2.print();

	MatrixF m3 = m1+m2;
	m3.print();

	MatrixF m4 = m3*3;
	m4.print();

	MatrixF m5 = m1*m2;
	m5.print();

	float mf6[9] = {1,2,3,4,5,6};
	MatrixF m6 = newMatrixFromRAM(2,3,mf6);
	VectorF v5 = m6*v1;
	v5.print();

	cout<<v1.length()<<endl;


	m6.print();
	m6.transpose().print();

	cudaDeviceSynchronize();
    return 0;
}
