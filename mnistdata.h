#ifndef MNISTDATA_H_INCLUDED
#define MNISTDATA_H_INCLUDED

#include <vector>
#include <numeric>
#include <random>
#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

template<typename T>
class MnistData{
    const std::vector<VectorXd> imgData;
    const std::vector<T> labelData;
    std::vector<unsigned int> idx;
public:
    const unsigned int size;
    MnistData(const std::vector<VectorXd> &iData, const std::vector<T> &lData);
    ~MnistData()=default;
    void shuffle();
    const VectorXd& getImg(const unsigned int n)const;
    const T& getLbl(const unsigned int n)const;
    const MatrixXd getImgBatch(const unsigned int startIdx, unsigned int mbsize)const;
    const MatrixXd getLblBatch(const unsigned int startIdx, unsigned int mbsize)const;

};

template<typename T>
MnistData<T>::MnistData(const std::vector<VectorXd> &iData, const std::vector<T> &lData): imgData(iData), labelData(lData), size(iData.size()){
    idx.resize(size);
    iota(idx.begin(), idx.end(), 0);
}
template<typename T>
void MnistData<T>::shuffle(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(idx.begin(), idx.end(), gen);
}
template<typename T>
const VectorXd& MnistData<T>::getImg(unsigned int n)const{
    return imgData[idx[n]];
}
template<typename T>
const T& MnistData<T>::getLbl(unsigned int n)const{
    return labelData[idx[n]];
}
template<typename T>
const MatrixXd MnistData<T>::getImgBatch(const unsigned int startIdx, unsigned int mbsize)const{
    MatrixXd mBatchImg;
    mbsize=(startIdx+mbsize<=size)? mbsize : size-startIdx;
    //std::cout << startIdx << " to " << startIdx+mbsize << std::endl;
    if(size>0){
        mBatchImg.resize(imgData[0].size(), mbsize);
        for(unsigned int n=0; n<mbsize; n++)
            mBatchImg.col(n)=imgData[idx[startIdx+n]];
    }
    return mBatchImg;

}
template<typename T>
const MatrixXd MnistData<T>::getLblBatch(const unsigned int startIdx, unsigned int mbsize)const{
    MatrixXd mBatchLbl;
    unsigned int rows=(typeid(T)==typeid(VectorXd))? 10 : 1;
    if(size>0 && typeid(T)==typeid(VectorXd)){
        mBatchLbl.resize(rows, mbsize);
        for(unsigned int n=0; n<mbsize; n++)
            mBatchLbl.col(n)=labelData[idx[startIdx+n]];

    }
    return mBatchLbl;
}

#endif // MNISTDATA_H_INCLUDED
