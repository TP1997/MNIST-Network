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

    vector<VectorXf> neuronInput;                                      //Neuron z (input) values.

    vector<VectorXf> neuronActivation;                                 //Neuron activation (output) values.
                                                                       //   neuronActivation[l].(n)     = n:th neuron in the l:th layer.
                                                                       //   neuronActivation[0]         = Activation values in input layer.
                                                                       //   neuronActivation[l]         = Activation values in hidden layer l (0 < l < mLayers-1).
                                                                       //   neuronActivation[mLayers-1] = Activation values in output layer.

//Member functions.
    VectorXf sigmoid(const VectorXf &z);                                                        //Sigmoid (activation) function.
    vector<VectorXf> feedforward(VectorXf l);                                                   //Feedforward.
    VectorXf feedforward2(VectorXf a);                                                          //Feedforward2.
    void SGD(MatrixXf miniBatch, unsigned int batchSize, float learningRate);                   //Learning algorithm.
    void SGD_update(const MatrixXf &minibatch, const MatrixXf &corrOutput, float learningRate); //update weights & biases.
    void backpropagate(const VectorXf &trnExp, const VectorXf &y);                              //Backpropagate.
    vector<VectorXf> backpropagate2(const VectorXf &trnExp, const VectorXf &y);                 //Backpropagate2.
    void backpropagate3(const VectorXf &trnExp, const VectorXf &y);                             //Backpropagate3.
    float cost(const VectorXf &y);                                                              //Network's cost function.
    VectorXf Dcost(const VectorXf &y);                                                          //Derivative of cost function (respect by output layer)
    VectorXf Dsigmoid(const VectorXf &z);                                                       //Derivative of sigmoid function.
};

#endif // NETWORK_H_INCLUDED
