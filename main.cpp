#include <iostream>
#include <fstream>
#include <numeric>
#include <cstdlib>
#include <ctime>
#include <algorithm>

#include "network.h"
#include "mnistdata.h"

using namespace std;
/** NOT USED FUNCTIONS                                    */
//Debugging:
bool tryParse(string &input, int &output);
void toTxtFile(const char *fname, const vector<double> &buffer);

/** USED FUNCTIONS                                          */
//Printing:
void printVectorImage(const VectorXd &vec, const unsigned int rows, const unsigned int cols);
void printVectorLabelConverted(const VectorXd &vec);
//Helpers:
void bytewiseReverseInt(uint32_t &n);
unsigned char* getSingleMnistImageData(unsigned const char *buffer, int idx, const unsigned int imgSize);
VectorXd toDoubleVector(unsigned char *buffer, const unsigned int bufferSize);
double getVectorLabelConverted(const VectorXd &vec);
//Initalization:
unsigned char* mnistImageReader(const char *path, unsigned int &rows, unsigned int &cols, unsigned int &noOfImages);
unsigned char *mnistLabelReader(const char *path);
vector<VectorXd> parseImageData(unsigned char *buffer, const unsigned int imgSize, const unsigned int noOfImages);
vector<VectorXd> parseLabelData(unsigned char *buffer, const unsigned int nOfElements);
vector<double> parseLabelData2(unsigned char *buffer, const unsigned int nOfElements);
//Training:
vector<unsigned int> pickRandomIdices(const unsigned int cnt, const unsigned int upperRange);
MatrixXd crtRndmMinibatch(vector<VectorXd> &data, vector<unsigned int> &idx, const unsigned int imgSize);
MatrixXd crtOutputVectors(vector<VectorXd> &data, vector<unsigned int> &idx);

/** GLOBALS                                                 */                              //Training-set, 60 000 examples.
const char* TRAINING_SET_IMAGE_FILE =    "mnist/training_set/train-images.idx3-ubyte";           //Training-set images.
const char* TRAINING_SET_LABEL_FILE =    "mnist/training_set/train-labels.idx1-ubyte";           //Training-set labels.
                                                                                            //Test-set, 10 000 examples.
const char* TEST_SET_IMAGE_FILE     =    "mnist/test_set/t10k-images.idx3-ubyte";                //Test-set images.
const char* TEST_SET_LABEL_FILE     =    "mnist/test_set/t10k-labels.idx1-ubyte";                //Test-set labels.

/************************************************************/
/***************** MAIN FUNCTION START **********************/
int main()
{
//                      VARIABLE INITALIZATION.
    unsigned int rows, cols;                                                                //Image dimensions
    unsigned int trnCount, tstCount=0;                                                      //Number of images in the set.
//Load training set.
    unsigned char *trnData  = mnistImageReader(TRAINING_SET_IMAGE_FILE, rows, cols, trnCount);
    unsigned char *trnLabel = mnistLabelReader(TRAINING_SET_LABEL_FILE);
//Load test set.
    unsigned char *tstData   = mnistImageReader(TEST_SET_IMAGE_FILE, rows, cols, tstCount);
    unsigned char *tstLabel  = mnistLabelReader(TEST_SET_LABEL_FILE);
//Parse & convert training data.
    vector<VectorXd> imageTrnData = parseImageData(trnData, rows*cols, trnCount);
    vector<VectorXd> labelTrnData = parseLabelData(trnLabel, trnCount);
//Parse & convert test data.
    vector<VectorXd> imageTstData  = parseImageData(tstData, rows*cols, tstCount);
    vector<double> labelTstData  = parseLabelData2(tstLabel, tstCount);
//Create training set.
    MnistData<VectorXd> trainingData(imageTrnData, labelTrnData);
//Create test set.
    MnistData<double> testData(imageTstData, labelTstData);
//Create network.


//                  NETWORK TRAINING
    unsigned int epochs = 1;                                                            //Number of training ephocs.
    unsigned int mbSize = 10;                                                           //Size of SGD training batch.
    double eta          = 3.0;                                                          //Network's learning rate.
    vector<double> NN_accuracy;                                                         //Get accuracy results while training.

    Network network(3, 784, 30, 10);                                                    //Create network with 3 layers with sizes of 784, 30, 10.

    //network.SGD_debug(imageTrainData, labelTrainData, mbSize, epochs, eta, imageTestData, labelTestData, NN_accuracy);
    //toTxtFile("accuracy.txt", NN_accuracy);
    network.train(trainingData, mbSize, epochs, eta);                             //New
    unsigned int rec=network.evaluate(testData);
    cout << rec << "/" << tstCount << endl;

//                  NETWORK TESTSPACE

    bool quit=false;
    string input;
    int idx=0;
    VectorXd outputVec(10);

    cout << "Type index in range [0, " << tstCount-1 << "]. Quit by typing some rubbish." << endl;
    getline(cin, input);
    quit=tryParse(input, idx);
    while(!quit){
        printVectorImage(testData.getImg(idx), rows, cols);
        outputVec=network.recDigit(testData.getImg(idx));
        cout << "Correct value: " << testData.getLbl(idx) << endl;
        cout << "Network recognized as: " << getVectorLabelConverted(outputVec) << endl;
        //cout << "Networks output as vector:\n" << outputVec << endl;

        cout << "Type index in range [0, " << tstCount-1 << "]. Quit by typing some rubbish." << endl;
        getline(cin, input);
        quit=tryParse(input, idx);

    }
    cout << "End..." << endl;

    return 0;
}
/************************************************************/
/**************** FUNCTION DECLARATIONS (USED) **************/


/** Read image data stored in binary file.
*   @param *path Filepath to binary file.
*   @param rows Reference.
*   @param cols Reference.
*   @param noOfImagesReference.
*   @return buffer Whole image data set stored in byte array.
*/
unsigned char* mnistImageReader(const char *path, unsigned int &rows, unsigned int &cols, unsigned int &nOfImages){
    unsigned char *buffer;
    uint32_t magicNumber=0;

    ifstream inputFile(path, ios_base::binary);
    if(inputFile.is_open()){
    //Magic number.
        inputFile.read((char*)&magicNumber, sizeof(magicNumber));
        bytewiseReverseInt(magicNumber);
   //Number of images.
        inputFile.read((char*)&nOfImages, sizeof(nOfImages));
        bytewiseReverseInt(nOfImages);
    //Number of pixel rows in image.
        inputFile.read((char*)&rows, sizeof(rows));
        bytewiseReverseInt(rows);
    //Number of pixel columns in image.
        inputFile.read((char*)&cols, sizeof(cols));
        bytewiseReverseInt(cols);

    //Allocate space for image-data-buffer.
        buffer=new unsigned char[nOfImages*rows*cols];
    //Read image data to buffer.
        inputFile.read((char*)&buffer[0], (nOfImages*rows*cols)*sizeof(char));

        inputFile.close();
        return buffer;
    }
    else{
        cout << "Can't open the file." << endl;
        return nullptr;
    }
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
        bytewiseReverseInt(magicNumber);
    //Number of items.
        inputFile.read((char*)&noOfItems, sizeof(noOfItems));
        bytewiseReverseInt(noOfItems);

    //Allocate space for label-data-buffer.
        buffer=new unsigned char[noOfItems];
    //Read label data to buffer.
        inputFile.read((char*)&buffer[0], (noOfItems)*sizeof(char));

        inputFile.close();
        return buffer;
    }
    else{
        cout << "Can't open the file." << endl;
        return nullptr;
    }

}

/** Parse MNIST-image-data and convert it to network's input format.
*   @param buffer Mnist image data.
*   @return images, Set of Mnist-images in vector format.
*/
vector<VectorXd> parseImageData(unsigned char *buffer, const unsigned int imgSize, const unsigned int nOfImages){
    vector<VectorXd> images;
    images.reserve(nOfImages);

    for(unsigned int n=0; n<nOfImages; n++){
        unsigned char *imgData=getSingleMnistImageData(buffer, n, imgSize);
        images.push_back(toDoubleVector(imgData, imgSize));
    }

    return images;
}

/** Parse Mnist-label-data and convert it to network's input format.
*   @param buffer, Mnist label data.
*   @param nOfElements, Number of individual elements in label data.
*   @return labels, Set of Mnist-labels in vector format.
*/
vector<VectorXd> parseLabelData(unsigned char *buffer, const unsigned int nOfElements){
    vector<VectorXd> labels(nOfElements);

    unsigned int n=0;
    for_each( labels.begin(), labels.end(), [&](VectorXd &v){v.resize(10); v(int(buffer[n++]))=1.0;} );
    //for_each( labels.begin(), labels.end(), [&](VectorXd &v){v.resize(10); v.setOnes(10); v*=0.01; v(int(buffer[n++]))=0.99f;} );

    return labels;
}

/**
*
*/
vector<double> parseLabelData2(unsigned char *buffer, const unsigned int nOfElements){
    vector<double> labels(nOfElements);

    unsigned int i=0;
    for_each( labels.begin(), labels.end(), [&](double &d){d=(double)buffer[i++];} );

    return labels;
}

/** Pick random indices in range [0, upperRange-1].
*   @param cnt, Number of indeces generated.
*   @param upperRange
*   @return indices, Array of random indices.
*/
vector<unsigned int> pickRandomIdices(const unsigned int cnt, const unsigned int upperRange){
    vector<unsigned int> indices(cnt);

    srand(rndmSeed());
    for_each( indices.begin(), indices.end(), [&upperRange](unsigned int &i){i=(rand()%upperRange);} );

    return indices;
}

/** Function for creating random mini-batch.
*   @param data, Mnist-image-data in network-input-format.
*   @param idx, Vector of randomly picked index values.
*   @param imgSize, Size of single Mnist-image (number of pixels).
*   @return batch, Mini-batch stored in matrix-form. Each column represents data of single Mnist-image.
*/
MatrixXd crtRndmMinibatch(vector<VectorXd> &data, vector<unsigned int> &idx, const unsigned int imgSize){
    MatrixXd batch(imgSize, idx.size());
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
MatrixXd crtOutputVectors(vector<VectorXd> &data, vector<unsigned int> &idx){
    MatrixXd labels(10, idx.size());
    for(unsigned int n=0; n<idx.size(); n++){
        labels.col(n)=data[idx[n]];
    }
    return labels;
}

/** Reverse 32-bit unsigned integer bytewise
*   @param &n, Integer to be reversed.
*/
void bytewiseReverseInt(uint32_t &n){
    n=((n>>16)&0x0000ffff) | ((n<<16)&0xffff0000);
    n=((n>>8)&0x00ff00ff) | ((n<<8)&0xff00ff00);
}

/** Get data of single image from image-byte array.
*   @param buffer, Pointer to buffer which holds the overall image data.
*   @param idx, Starting index of image.
*   @param imgSize, Size of single image.
*   @return bufferCopy, Image data stored in byte array.
*/
unsigned char* getSingleMnistImageData(unsigned const char *buffer, int idx, const unsigned int imgSize){
    unsigned char *bufferCopy=new unsigned char[imgSize];
    memcpy(bufferCopy, &buffer[idx*imgSize], imgSize+1);
    return bufferCopy;
}

/** Function that converts char-to-double and normalize value to range [0,1].
*   @param buffer, Array that holds the convertable data.
*   @param buferSize, Size of convertable bufer.
*   @return vec, double vector that holds converted values.
*/
VectorXd toDoubleVector(unsigned char *buffer, const unsigned int bufferSize){
    VectorXd vec(bufferSize);

    unsigned int n=0;
    for_each( vec.data(), vec.data()+vec.size(), [&](double &d){d=buffer[n++]/255.0;} );//*(0.99/255.0)+0.01

    return vec;
}

/** Function for printing (Test/Training) image.
*   @param vec, Image's data in vector-format.
*   @param rows, Image's height (pixels).
*   @param cols, Image's width  (pixels).
*/
void printVectorImage(const VectorXd &vec, const unsigned int rows, const unsigned int cols){
    double minE=vec.minCoeff();
    for(unsigned int n=0; n<vec.rows(); n++){
        printf("%c%s", (vec(n)==minE? '.':'#'), ((n+1)%rows==0)?( ((n+1)%(rows*cols)==0)? "\n\n" : "\n" ) : " ");
    }
}

/** Function for printing vector-format label in numeric format.
*   @param vec, Label data in vector format.
*/
void printVectorLabelConverted(const VectorXd &vec){
    unsigned int maxIdx;
    vec.maxCoeff(&maxIdx);
    cout << maxIdx << endl;
}

/*
*
*
*/
double getVectorLabelConverted(const VectorXd &vec){
    unsigned int maxIdx;
    vec.maxCoeff(&maxIdx);

    return maxIdx;
}

/** DEBUGGING
*
*/
bool tryParse(string &input, int &output){
    try{
        output=stoi(input);
    }
    catch(std::invalid_argument &e){
        return true;
    }
    return false;

}
void toTxtFile(const char *fname, const vector<double> &buffer){
    ofstream file(fname);
    for(double d : buffer){
        file << d << "\n";
    }
    file.close();
}
