#include <iostream>
#include <fstream>

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
void printMnist(unsigned char *buffer, const unsigned int rows, const unsigned int cols, const unsigned int noOfImages);
unsigned char* getSingleMnistImageData(unsigned const char *buffer, int idx, const size_t imgSize);
VectorXf charToFloatVector(const unsigned char *buffer, const unsigned int bufferSize);

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
    unsigned int rows, cols, noOfImages;
    unsigned char *data=mnistImageReader(TEST_SET_IMAGE_FILE, rows, cols, noOfImages);
    unsigned char *labels=mnistLabelReader(TEST_SET_LABEL_FILE);
    //printMnist(data, rows, cols, 1);

    unsigned char *img1=getSingleMnistImageData(data, 0, rows*cols*sizeof(char));
    VectorXf vec=charToFloatVector(img1, rows*cols);
    for(unsigned int n=0; n<vec.rows(); n++){
        //const char *eol=(n%rows==0)?( (n%(rows*cols)==0)? "\n\n" : "\n" ) : " ";
        //cout << (buffer[n]==0? ".":"#") << eol;
        printf("%c%s", (vec(n)==0.0f? '.':'#'), ((n+1)%rows==0)?( ((n+1)%(rows*cols)==0)? "\n\n" : "\n" ) : " ");
    }
    //unsigned char trueVal=labels[1];
    //cout << "Number: " << (int)trueVal << endl;
    //printMnist(img1, rows, cols, 1);
    //cout << "from buffer" << endl;
    //printMnist(data, rows, cols, noOfImages);


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

/** Get data of single image from image-byte array.
*   @param buffer Pointer to buffer which holds the overall image data.
*   @param idx Starting index of image.
*   @param imgSize Size of single image.
*   @return bufferCopy Image data stored in byte array.
*/
unsigned char* getSingleMnistImageData(unsigned const char *buffer, int idx, const size_t imgSize){
    unsigned char *bufferCopy=new unsigned char[imgSize];
    memcpy(bufferCopy, &buffer[idx*imgSize], imgSize+1);
    return bufferCopy;
}

/** Print images from Mnist-dataset
*   @param buffer Pointer to buffer which holds the image data.
*   @param rows Pixel rows in single image.
*   @param cols Pixel columns in single image.
*   @param noOfImages Number of images which will be printed.
*/
void printMnist(unsigned char *buffer, const unsigned int rows, const unsigned int cols, const unsigned int noOfImages){
    for(unsigned int n=0; n<noOfImages*rows*cols; n++){
        //const char *eol=(n%rows==0)?( (n%(rows*cols)==0)? "\n\n" : "\n" ) : " ";
        //cout << (buffer[n]==0? ".":"#") << eol;
        printf("%c%s", (buffer[n]==0? '.':'#'), ((n+1)%rows==0)?( ((n+1)%(rows*cols)==0)? "\n\n" : "\n" ) : " ");
    }
}

/** Reverse 32-bit unsigned intefer bytewise
*   @param n Integer to reversed.
*   @return n Bytewise-reversed integer.
*/
uint32_t bytewiseReverseInt(uint32_t n){
    n=((n>>16)&0x0000ffff) | ((n<<16)&0xffff0000);
    n=((n>>8)&0x00ff00ff) | ((n<<8)&0xff00ff00);
    return n;
}

/**
*
*
*/
VectorXf charToFloatVector(const unsigned char *buffer, const unsigned int bufferSize){
    VectorXf vec(bufferSize);
    for(unsigned int n=0; n<bufferSize; n++){
        vec(n)=buffer[n]/255.0f;
    }
    return vec;
}

/** TRASH                   */
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
