#ifndef CUML_LEARN_MNIST_LOADER_H
#define CUML_LEARN_MNIST_LOADER_H

#include <vector>
#include <fstream>
#include <LA.h>


using namespace std;


static int to_int(char* p)
{
    return ((p[0] & 0xff) << 24) | ((p[1] & 0xff) << 16) |
           ((p[2] & 0xff) <<  8) | ((p[3] & 0xff) <<  0);
}

static vector<int> load_label(string path){
    ifstream file(path,ios::in|ios::binary);
    if(file){
        file.seekg(0,ios::end);
        auto size=file.tellg();
        file.seekg(0,ios::beg);

        char keyData[4];
        file.read(keyData,4);
        int key=to_int(keyData);
        if(key!=0x801){
            cerr<<"incorrect file"<<endl;
        }
        else{

            char imageCountData[4];
            file.read(imageCountData,4);
            int imageCount=to_int(imageCountData);

            if(imageCount>50000) imageCount=50000;

            vector<int> result;

            for (int i = 0; i <imageCount ; ++i) {
                char thisImage[1];
                file.read(thisImage,1);
                result.emplace_back(thisImage[0]);
            }
            file.close();
            return result;
        }
    }
};


static vector<VectorF> load_image(string path){
    ifstream file(path,ios::in|ios::binary);
    if(file){
        file.seekg(0,ios::end);
        auto size=file.tellg();
        file.seekg(0,ios::beg);

        char keyData[4];
        file.read(keyData,4);
        int key=to_int(keyData);
        if(key!=0x803){
            cerr<<"incorrect file"<<endl;
        }
        else{

            char imageCountData[4];
            file.read(imageCountData,4);
            int imageCount=to_int(imageCountData);

            if(imageCount>50000) imageCount=50000;

            char rowsData[4];
            file.read(rowsData,4);
            int rows=to_int(rowsData);

            char colsData[4];
            file.read(colsData,4);
            int cols=to_int(colsData);

            int imageSize=rows*cols;

            vector<VectorF> result;

            float* singleImageData = malloc(imageSize* sizeof(*singleImageData));

            for (int i = 0; i <imageCount ; ++i) {
                char thisImage[imageSize];
                file.read(thisImage,imageSize);
                for (int pixel = 0; pixel < imageSize; ++pixel) {
                    unsigned char thisPixelData=thisImage[pixel];
                    float thisPixel= (float)thisPixelData/255.f;
                    singleImageData[pixel]=thisPixel;
                }
                result.emplace_back(newVectorFromRAM(imageSize,singleImageData));
            }
            file.close();
            free(singleImageData);
            return result;
        }
    }
};


#endif //CUML_LEARN_MNIST_LOADER_H
