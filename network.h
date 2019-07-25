#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;

typedef Matrix<float, Dynamic, 1> VectorXf;
typedef Matrix<float, Dynamic, Dynamic> MatrixXf;

class Network{
public:
    Network(int layers, ...);
    ~Network();

private:
//Member variables.
    unsigned int mLayers;                                              //Count of layers in network.

    unsigned int mNeurons;                                             //Count of neurons in network.

    vector<unsigned int> aSizes;                                       //Sizes for individual layers.
                                                                       //   aSizes[l]                   = neuron count for l:th layer.

    vector<VectorXf> biases;                                           //Vector of network's bias values for each layer. There are no bias values for first (input) layer
                                                                       //   biases[l]                   = bias values for l+1:th layer.
                                                                       //   ==> biases[0]=layer 1, biases[1]=layer 2 ...
                                                                       //   biases[l].(n)               = bias value of n:th neuron in l+1:th layer,

    vector<MatrixXf> weightMatrices;                                   //Listing of weight matrices used for individual layers.
                                                                       //   Each matrix is constructed by M(row, col).
                                                                       //   weightMatrices[l]           = weight matrix for l+1:th layer.
                                                                       //   ==> l-1 - weightMatrices[l] - l.

    vector<VectorXf> neuronActivation;                                 //Neuron activation (output) values.
                                                                       //   neuronActivation[l].(n)     = n:th neuron in the l:th layer.
                                                                       //   neuronActivation[0]         = Activation values in input layer.
                                                                       //   neuronActivation[l]         = Activation values in hidden layer l (0 < l < mLayers-1).
                                                                       //   neuronActivation[mLayers-1] = Activation values in output layer.

//Member functions.
    VectorXf sigmoid(const VectorXf &z);                                     //Sigmoid (activation) function.
    VectorXf Dsigmoid(const VectorXf &z);                                    //Derivative of sigmoid function.
    void feedforward(VectorXf l);
    VectorXf feedforward2(VectorXf a);
    void SGD();
};

#endif // NETWORK_H_INCLUDED
