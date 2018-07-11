#ifndef CUML_LEARN_NETWORK_H
#define CUML_LEARN_NETWORK_H

#include <vector>
#include <cmath>
#include <iostream>
#include <random>

using namespace std;

class Network{
public:
    vector<MatrixF> weights;
    vector<VectorF> bias;
    vector<int> size;
    string name;
    Network(const vector<int>&& sizes):size(sizes){

        std::random_device rd{};
        std::mt19937 gen{rd()};

        // values near the mean are the most likely
        // standard deviation affects the dispersion of generated values from the mean

        std::normal_distribution<> biasDistribution{0,1};

        for (int i = 0; i <sizes.size()-1 ; ++i) {


            float* weightData = malloc( sizes[i+1]*sizes[i]*sizeof(*weightData));
            float* biasData = malloc(sizes[i+1]*sizeof(*biasData));

            std::normal_distribution<> weightDistribution{0,1/sqrt(sizes[i])};

            for(int w = 0;w<sizes[i+1]*sizes[i];++w){
            	weightData[w] =  weightDistribution(gen);
            }

            for(int b = 0;b<sizes[i+1];++b){
            	biasData = biasDistribution(gen);
            }

            weights.emplace_back(newMatrixFromRAM(sizes[i+1],sizes[i],weightData));
            bias.emplace_back(newVectorFromRAM(sizes[i+1],biasData));

            free(weightData);
            free(biasData);
        }
    }


    inline VectorF cost_delta(VectorF z, VectorF a,VectorF y ){
        return (a-y);//cross-entropy cost
        return (a-y).cwiseProduct(sigmoid_prime(z));//quadratic cost
    }




    inline VectorF sigmoid(VectorF input){
        for (int i = 0; i < input.size(); ++i) {
            input[i]= 1.0/(1.0+exp(-input[i]));
        }
        return input;
    }

    inline VectorF sigmoid_prime(VectorF input){
        for (int i = 0; i < input.size(); ++i) {
            double sig=1.0/(1.0+exp(-input[i]));
            input[i]= sig*(1.0-sig);
        }
        return input;
    }

    inline VectorF feedForward(VectorF input){
        for (int i = 0; i < size.size()-1; ++i) {
            input= weights[i]*input;
            input=input+bias[i];
            input= sigmoid(input);
        }
        return input;
    }

    void train(vector<VectorF>& images,vector<int>& labels,int epochs,int batch_size,double eta,double lambda){
        for (int i = 0; i <epochs ; ++i) {
            auto time=std::time(0);
            std::default_random_engine generatorW(time);
            std::default_random_engine generatorB(time);
            shuffle(images.begin(),images.end(),generatorW);
            shuffle(labels.begin(),labels.end(),generatorB);
            for (int j = 0; j <images.size() ; j+=batch_size) {
                train_batch(images,labels,j,min(batch_size,(int)images.size()-j),eta,lambda);
                //cout<<"finished epoch "<<i<<"    :   batch "<<j/batch_size<<endl;
            }
            cout<<"finished epoch "<<i<<endl;
        }
    }

    inline VectorF getCorrectOutput (int label){
        VectorF result=VectorF::Zero(10);
        result[label]=1;
        return result;
    }

    void train_batch(const vector<VectorF>& images,const vector<int>& labels,int start,int batch_size,double eta,double lambda){
        vector<MatrixF> nablaW;
        vector<VectorF> nablaB;
        for (int i = 0; i <size.size()-1 ; ++i) {
            nablaW.emplace_back(MatrixF::Zero(size[i+1],size[i]));
            nablaB.emplace_back(VectorF::Zero(size[i+1]));
        }

        MatrixF& w0=weights[0];
        MatrixF& w1=weights[1];
        VectorF& b0=bias[0];
        VectorF& b1=bias[1];
        MatrixF& nw0=nablaW[0];
        MatrixF& nw1=nablaW[1];
        VectorF& nb0=nablaB[0];
        VectorF& nb1=nablaB[1];

        for (int i = start; i < start+batch_size; ++i) {
            auto gradient=backPropogation(images[i],labels[i]);

            for (int j = 0; j <nablaW.size() ; ++j) {
                nablaW[j]=nablaW[j]+gradient.first[j];
                nablaB[j]= nablaB[j]+gradient.second[j];
            }

        }

        for (int i = 0; i <nablaW.size() ; ++i) {
            weights[i]=(1.0-(eta*lambda/(double)images.size()))*weights[i]-nablaW[i]*eta/(double)batch_size;
            bias[i]=bias[i]-nablaB[i]*eta/(double)batch_size;
        }

        return;

    }

    pair<vector<MatrixF>,vector<VectorF>> backPropogation(VectorF input,int label){
        vector<MatrixF> nablaW;
        vector<VectorF> nablaB;
        for (int i = 0; i <size.size()-1 ; ++i) {
            nablaW.emplace_back(MatrixF::Zero(size[i+1],size[i]));
            nablaB.emplace_back(VectorF::Zero(size[i+1]));
        }
        VectorF activation = input;
        vector<VectorF> activations={activation};
        vector<VectorF> zs;
        for (int i = 0; i < size.size()-1; ++i) {
            VectorF z=weights[i]*activation;
            z=z+bias[i];
            zs.push_back(z);
            activation=sigmoid(z);
            activations.push_back(activation);
        }
        VectorF correctOutput=getCorrectOutput(label);

        VectorF delta=cost_delta(zs.back(),activation,correctOutput);
        nablaB.back()=delta;
        nablaW.back()=columnMatrix(delta)*rowMatrix(activations[activations.size()-2]);


        for (int i = size.size()-3; i >=0 ; --i) {
            delta=sigmoid_prime(zs[i]).cwiseProduct( weights[i+1].transpose()*delta );
            nablaB[i]=delta;
            nablaW[i]=columnMatrix(delta)*rowMatrix(activations[i]);
        }
        return{nablaW,nablaB};

    };

    void test(const vector<VectorF>& images,const vector<int>& labels){
        int correct=0;
        for (int i = 0; i < images.size(); ++i) {
            auto result=feedForward(images[i]);
            if(result[labels[i]]>0.5) ++correct;
        }
        cout<<"Total: "<<images.size()<<endl;
        cout<<"Correct: "<<correct<<endl;
        cout<<"Percentage: "<<(double)correct/(double)images.size()<<endl;
    }

    void testSingle(VectorF image){
        auto result=feedForward(image);
        for (int i = 0; i <10 ; ++i) {
            cout<<i<<":  "<<result[i]<<endl;
        }
    }

    inline MatrixF columnMatrix(VectorF vec){
        MatrixF result(vec.size(),1);
        for (int i = 0; i <vec.size() ; ++i) {
            result(i,0)=vec[i];
        }
        return result;
    }
    inline MatrixF rowMatrix(VectorF vec){
        return columnMatrix(vec).transpose();
    }


};

static inline void debug(){
    int i=0;
    i++;
}

#endif //CUML_LEARN_NETWORK_H
