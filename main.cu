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
#include "NN/Network.h"
#include "NN/MNIST_Loader.h"
using namespace std;

cublasHandle_t cublas_handle = nullptr;
int dataCount = 0;


int main( void ) {
	initCUDA();

	string command;
	Network network({28*28,30,10});
	cout<<"loading"<<endl;
	string dir="/Users/AmesingFlank/cuda-workspace/cuML_Learn/Debug/";
	auto train_image=load_image(dir+"train-images-idx3-ubyte");
	auto train_label=load_label(dir+"train-labels-idx1-ubyte");
	auto test_image=load_image(dir+"t10k-images-idx3-ubyte");
	auto test_label=load_label(dir+"t10k-labels-idx1-ubyte");
	while(command!="stop"){
		cout<<"please input command"<<endl;
		cin>>command;
		if(command=="train"){
			int epoch,batch_size;
			double eta,lambda;
			cin>>epoch>>batch_size>>eta>> lambda;
			network.train(train_image,train_label,epoch,batch_size,eta,lambda);
		}
		else if(command=="test"){
			network.test(test_image,test_label);
		}
	}
    return 0;
}
