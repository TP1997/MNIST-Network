#include <iostream>
#include <fstream>
#include <numeric>
#include <cstdlib>
#include <ctime>
#include <algorithm>

#include <SFML/Graphics/Image.hpp>

#include "network.h"

using namespace std;
/** FUNCTION DEFINITIONS                                    */
//Trash
unsigned char *loadImageData(const char *path, unsigned int &width, unsigned int &height, bool flip=true);
int reverseInt2 (int i);
char reverseChar(char c);
void printBit(int n);
//Used
unsigned char *mnistImageReader(const char *path, unsigned int &rows, unsigned int &cols, unsigned int &noOfImages);
unsigned char *mnistLabelReader(const char *path);
uint32_t reverseInt(uint32_t n);
uint32_t bytewiseReverseInt(uint32_t n);
unsigned char* getSingleMnistImageData(unsigned const char *buffer, int idx, const size_t imgSize);
VectorXf toFloatVector(unsigned char *buffer, const unsigned int bufferSize);
VectorXf trainingExampleOutput(unsigned char *buffer, const unsigned int idx);

MatrixXf createRandomMiniBatch(unsigned char *mnistBuffer, unsigned int noOfImages, const unsigned int batchSize, const unsigned int imagePixelCount);
//Printing & debugging.
void printMnist(unsigned char *buffer, const unsigned int rows, const unsigned int cols, const unsigned int noOfImages);
void printVectorImage(const VectorXf &vec, const unsigned int rows, const unsigned int cols);
void printVectorLabel(const VectorXf &vec);
void printVectorLabelConverted(const VectorXf &vec);

//Functions for parsing the MNIST-data and transforming it to type that network accepts.
vector<VectorXf> parseImageData(unsigned char *buffer, const size_t imgSize, const unsigned int noOfImages);
vector<VectorXf> parseLabelData(unsigned char *buffer, const unsigned int noOfElements);
//Function for picking random indices for sampling the MNIST-data.
vector<unsigned int> pickRandomIdices(unsigned int cnt, unsigned int upperRange);
//Function for creating random mini-batch.
MatrixXf crtRndmMinibatch(vector<VectorXf> &data, vector<unsigned int> &idx, const unsigned int imgSize);
//Function for picking creating correct outputs based on created minibatch.
MatrixXf crtOutputVectors(vector<VectorXf> &data, vector<unsigned int> &idx);

/** GLOBALS                                                 */                              //Training-set, 60 000 examples.
const char* TRAINING_SET_LABEL_FILE=    "mnist/training_set/train-labels.idx1-ubyte";           //Training-set labels.
const char* TRAINING_SET_IMAGE_FILE=    "mnist/training_set/train-images.idx3-ubyte";           //Training-set images.
                                                                                            //Test-set, 10 000 examples.
const char* TEST_SET_LABEL_FILE=        "mnist/test_set/t10k-labels.idx1-ubyte";                //Test-set labels.
const char* TEST_SET_IMAGE_FILE=        "mnist/test_set/t10k-images.idx3-ubyte";                //Test-set images.

/************************************************************/
/***************** MAIN FUNCTION START **********************/
int main()
{
    unsigned int rows, cols;
    unsigned int nOfImages;
    unsigned int batchSize=2;

//Load test set data.
    unsigned char *testData=mnistImageReader(TEST_SET_IMAGE_FILE, rows, cols, nOfImages);
    unsigned char *testLabels=mnistLabelReader(TEST_SET_LABEL_FILE);
//Parse & convert
    vector<VectorXf> imageTestData=parseImageData(testData, rows*cols, nOfImages);
    vector<VectorXf> desiredOutput=parseLabelData(testLabels, nOfImages);

//Pick random set of indices to sample images & output vectors.
    vector<unsigned int> rndmIdx=pickRandomIdices(batchSize, nOfImages);
//Create random minibatch.
    //MatrixXf minibatch=crtRndmMinibatch(imageTestData, rndmIdx, rows*cols);
//Create correct outputs based on minibatch.
    MatrixXf labels=crtOutputVectors(desiredOutput, rndmIdx);

//SANDBOX....

    for(unsigned int n=0; n<batchSize; n++){
        cout << "label :" << n << endl;
        printVectorLabel(labels.col(n));
        cout << endl;
        printVectorLabelConverted(labels.col(n));
        //printVectorImage(minibatch.col(n), rows, cols);

    }

    return 0;
}
/************************************************************/
/**************** FUNCTION DECLARATIONS *********************/

/** Read image data stored in binary file.
*   @param *path Filepath to binary file.
*   @param rows Reference.
*   @param cols Reference.
*   @param noOfImagesReference.
*   @return buffer Whole image data set stored in byte array.
*/
unsigned char* mnistImageReader(const char *path, unsigned int &rows, unsigned int &cols, unsigned int &noOfImages){
    unsigned char *buffer;
    uint32_t magicNumber=0;

    ifstream inputFile(path, ios_base::binary);
    if(inputFile.is_open()){
    //Magic number.
        inputFile.read((char*)&magicNumber, sizeof(magicNumber));
        magicNumber=bytewiseReverseInt(magicNumber);
   //Number of images.
        inputFile.read((char*)&noOfImages, sizeof(noOfImages));
        noOfImages=bytewiseReverseInt(noOfImages);
    //Number of pixel rows in image.
        inputFile.read((char*)&rows, sizeof(rows));
        rows=bytewiseReverseInt(rows);
    //Number of pixel columns in image.
        inputFile.read((char*)&cols, sizeof(cols));
        cols=bytewiseReverseInt(cols);

    //Allocate space for image-data-buffer.
        buffer=new unsigned char[noOfImages*rows*cols];
    //Read image data to buffer.
        inputFile.read((char*)&buffer[0], (noOfImages*rows*cols)*sizeof(char));

        inputFile.close();
    }
    else{
        cout << "Can't open file." << endl;
    }
    return buffer;
}

/** Read label data from binary file
*   @param path Filepath to binary file.
*   @return buffer Whole label data set stored in byte array.
*/
unsigned char *mnistLabelReader(const char *path){
    unsigned char *buffer;
    uint32_t magicNumber=0;
    uint32_t noOfItems=0;

    ifstream inputFile(path, ios_base::binary);
    if(inputFile.is_open()){
    //Magic number.
        inputFile.read((char*)&magicNumber, sizeof(magicNumber));
        magicNumber=bytewiseReverseInt(magicNumber);
    //Number of items.
        inputFile.read((char*)&noOfItems, sizeof(noOfItems));
        noOfItems=bytewiseReverseInt(noOfItems);

    //Allocate space for label-data-buffer.
        buffer=new unsigned char[noOfItems];
    //Read label data to buffer.
        inputFile.read((char*)&buffer[0], (noOfItems)*sizeof(char));

        inputFile.close();
    }
    else{
        cout << "Can't open file." << endl;
    }
    return buffer;

}

/** Parse MNIST-image-data and convert it to network's input format.
*   @param buffer Mnist image data.
*   @return images, Set of Mnist-images in vector format.
*/
vector<VectorXf> parseImageData(unsigned char *buffer, const size_t imgSize, const unsigned int noOfImages){
    vector<VectorXf> images;
    for(unsigned int n=0; n<noOfImages; n++){
        unsigned char *imgData=getSingleMnistImageData(buffer, n, imgSize);
        images.push_back(toFloatVector(imgData, imgSize));
    }
    return images;
}

/** Parse Mnist-label-data and convert it to network's input format.
*   @param buffer, Mnist label data.
*   @param nOfElements, Number of individual elements in label data.
*   @return labels, Set of Mnist-labels in vector format.
*/
vector<VectorXf> parseLabelData(unsigned char *buffer, const unsigned int nOfElements){
    vector<VectorXf> labels;
    for(unsigned int n=0; n<nOfElements; n++){
        unsigned int label=buffer[n];
        VectorXf y(10);
        y.setZero(10);
        y(label)=1.0f;
        labels.push_back(y);
    }
    return labels;
}

/** Get data of single image from image-byte array.
*   @param buffer, Pointer to buffer which holds the overall image data.
*   @param idx, Starting index of image.
*   @param imgSize, Size of single image.
*   @return bufferCopy, Image data stored in byte array.
*/
unsigned char* getSingleMnistImageData(unsigned const char *buffer, int idx, const size_t imgSize){
    unsigned char *bufferCopy=new unsigned char[imgSize];
    memcpy(bufferCopy, &buffer[idx*imgSize], imgSize+1);
    return bufferCopy;
}

/** Print images from Mnist-dataset
*   @param buffer, Pointer to buffer which holds the image data.
*   @param rows, Pixel rows in single image.
*   @param cols, Pixel columns in single image.
*   @param nOfImages, Number of images which will be printed.
*/
void printMnist(unsigned char *buffer, const unsigned int rows, const unsigned int cols, const unsigned int nOfImages){
    for(unsigned int n=0; n<nOfImages*rows*cols; n++){
        //const char *eol=(n%rows==0)?( (n%(rows*cols)==0)? "\n\n" : "\n" ) : " ";
        //cout << (buffer[n]==0? ".":"#") << eol;
        printf("%c%s", (buffer[n]==0? '.':'#'), ((n+1)%rows==0)?( ((n+1)%(rows*cols)==0)? "\n\n" : "\n" ) : " ");
    }
}

/** Function for printing (Test/Training) image.
*   @param vec, Image's data in vector-format.
*   @param rows, Image's height (pixels).
*   @param cols, Image's width  (pixels).
*/
void printVectorImage(const VectorXf &vec, const unsigned int rows, const unsigned int cols){
    for(unsigned int n=0; n<vec.rows(); n++){
        printf("%c%s", (vec(n)==0.0f? '.':'#'), ((n+1)%rows==0)?( ((n+1)%(rows*cols)==0)? "\n\n" : "\n" ) : " ");
    }
}

/** Function for printing label in vector-format.
*   @param vec, Label data in vector format.
*/
void printVectorLabel(const VectorXf &vec){
    cout << vec << endl;
}

/** Function for printing vector-format label in numeric format.
*   @param vec, Label data in vector format.
*/
void printVectorLabelConverted(const VectorXf &vec){
    int n=-1;
    while(vec(++n)!=1);
    cout << n << endl;
}

/** Reverse 32-bit unsigned integer bytewise
*   @param n, Integer to be reversed.
*   @return n, Bytewise-reversed integer.
*/
uint32_t bytewiseReverseInt(uint32_t n){
    n=((n>>16)&0x0000ffff) | ((n<<16)&0xffff0000);
    n=((n>>8)&0x00ff00ff) | ((n<<8)&0xff00ff00);
    return n;
}

/** Function that converts char-to-float and normalize value to range [0,1].
*   @param buffer, Array that holds the convertable data.
*   @param buferSize, Size of convertable bufer.
*   @return vec, Float vector that holds converted values.
*/
VectorXf toFloatVector(unsigned char *buffer, const unsigned int bufferSize){
    VectorXf vec(bufferSize);
    for(unsigned int n=0; n<bufferSize; n++){
        vec(n)=buffer[n]/255.0f;
    }
    return vec;
}

/** Pick random indices in range [0, upperRange-1].
*   @param cnt, Number of indeces generated.
*   @param upperRange
*   @return indices, Array of random indices.
*/
vector<unsigned int> pickRandomIdices(unsigned int cnt, unsigned int upperRange){
    vector<unsigned int> indices(cnt);

    srand(unsigned(time(0)));
    for(unsigned int n=0; n<cnt; n++)
        indices[n]=(rand()%upperRange);

    return indices;
}

/** Function for creating random mini-batch.
*   @param data, Mnist-image-data in network-input-format.
*   @param idx, Vector of randomly picked index values.
*   @param imgSize, Size of single Mnist-image (number of pixels).
*   @return batch, Mini-batch stored in matrix-form. Each column represents data of single Mnist-image.
*/
MatrixXf crtRndmMinibatch(vector<VectorXf> &data, vector<unsigned int> &idx, const unsigned int imgSize){
    MatrixXf batch(imgSize, idx.size());
    for(unsigned int n=0; n<idx.size(); n++){
        batch.col(n)=data[idx[n]];
    }
    return batch;
}

/** Function for creating correct network outputs based on created minibatch.
*   @param data, Mnist-label-data in network-input-format.
*   @param idx, Vector of randomly picked index values (same values used in crtRndmMinibatch).
*   @return labels, Labels stored in matrix-form, Each column reperesents data of single (correct) output.
*/
MatrixXf crtOutputVectors(vector<VectorXf> &data, vector<unsigned int> &idx){
    MatrixXf labels(10, idx.size());
    for(unsigned int n=0; n<idx.size(); n++){
        labels.col(n)=data[idx[n]];
    }
    return labels;
}


/********************************************************************************/
/********************************* TRASH ****************************************/
/**
*
*
*/
MatrixXf createRandomMiniBatch(unsigned char *mnistBuffer, unsigned int noOfImages, const unsigned int batchSize, const unsigned int imagePixelCount){
    MatrixXf miniBatch(imagePixelCount, batchSize);

//Set seed for random generation operations.
    std::srand(unsigned(std::time(0)));
    for(unsigned int n=0; n<batchSize; n++){
        //std::random_shuffle(indices.begin(), indices.end());
        int rndmIdx=rand()%noOfImages;
        unsigned char *rndImgData=getSingleMnistImageData(mnistBuffer, rndmIdx, imagePixelCount);
        miniBatch.col(n)=toFloatVector(rndImgData, imagePixelCount);
        delete[] rndImgData;
    }

    return miniBatch;
}

unsigned char *loadImageData(const char *path, unsigned int &width, unsigned int &height, bool flip){
    sf::Image img;
    if(!img.loadFromFile(path)){
        cout << "Coud not load image: " << path << endl;
        return nullptr;
    }
    if(flip)
        img.flipVertically();

    sf::Vector2u measures=img.getSize();
    width=measures.x;
    height=measures.y;
    unsigned char *data=(unsigned char*)img.getPixelsPtr();

    return data;
}
void printBit(int n){
    for(int s=31, m=1; s>=0; s--, m++){
        cout << ((n>>s)&1);
        if(m%4==0)
            cout << " ";
    }
}
char reverseChar(char c){
    c=((c>>2)&0x33) | ((c<<2)&0xcc);
    c=((c>>1)&0x55) | ((c<<1)&0xaa);

    return c;
}
uint32_t reverseInt(uint32_t n){
    n=((n>>16)&0x0000ffff) | ((n<<16)&0xffff0000);
    n=((n>>8)&0x00ff00ff) | ((n<<8)&0xff00ff00);
    n=((n>>4)&0x0f0f0f0f) | ((n<<4)&0xf0f0f0f0);
    n=((n>>2)&0x33333333) | ((n<<2)&0xcccccccc);
    n=((n>>1)&0x55555555) | ((n<<1)&0xaaaaaaaa);
    return n;
}
