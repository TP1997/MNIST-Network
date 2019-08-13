#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include <vector>
#include <chrono>
#include <Eigen/Dense>

#define rndmSeed() std::chrono::system_clock::now().time_since_epoch().count()

using namespace std;
using namespace Eigen;

//typedef Matrix<double, Dynamic, 1> VectorXf;
//typedef Matrix<double, Dynamic, Dynamic> MatrixXd;

class Network{
public:
    Network(int layers, ...);
    ~Network();
    void SGD(const vector<VectorXd> &trainSet, const vector<VectorXd> &labelSet, unsigned int mbSize, unsigned int ephocs, double eta); //Learning algorithm.
    void SGD_update(const MatrixXd &minibatch, const MatrixXd &y, double eta);  //treenaa enempi/tee minisimulaatio           //update weights & biases.
    //void SGD_update11(const vector<VectorXd> &minibatch, const vector<VectorXd> &y, double eta);                          //update weights & biases11.
    void SGD_update2(const MatrixXd &minibatch, const MatrixXd &y, double eta);                                             //update weights & biases 2.
    void evaluate(const vector<VectorXd> &testSet, const vector<VectorXd> &labelSet);
    VectorXd getOutput();
    VectorXd recDigit(const VectorXd & input);
private:
//Member variables.
    unsigned int mLayers;                                              //Count of layers in network.

    unsigned int mNeurons;                                             //Count of neurons in network.

    vector<unsigned int> aSizes;                                       //Sizes for individual layers.
                                                                       //   aSizes[l]                   = neuron count for l:th layer.

    vector<VectorXd> biases;                                           //Vector of network's bias values for each layer. There are no bias values for first (input) layer
                                                                       //   biases[l]                   = bias values for l+1:th layer.
                                                                       //   ==> biases[0]=layer 1, biases[1]=layer 2 ...
                                                                       //   biases[l].(n)               = bias value of n:th neuron in l+1:th layer,

    vector<MatrixXd> weightMatrices;                                   //Listing of weight matrices used for individual layers.
                                                                       //   Each matrix is constructed by M(row, col).
                                                                       //   weightMatrices[l]           = weight matrix for l+1:th layer.
                                                                       //   ==> l-1 - weightMatrices[l] - l.

    vector<VectorXd> neuronInput;                                      //Neuron z (input) values.

    vector<VectorXd> neuronActivation;                                 //Neuron activation (output) values.
                                                                       //   neuronActivation[l].(n)     = n:th neuron in the l:th layer.
                                                                       //   neuronActivation[0]         = Activation values in input layer.
                                                                       //   neuronActivation[l]         = Activation values in hidden layer l (0 < l < mLayers-1).
                                                                       //   neuronActivation[mLayers-1] = Activation values in output layer.

//Member functions.
    vector<VectorXd> feedforward(VectorXd l);                                                   //Feedforward.
    vector<VectorXd> backpropagate2(const VectorXd &trnExp, const VectorXd &y);                 //Backpropagate2.
    VectorXd Dcost(const VectorXd &y);                                                          //Derivative of cost function (respect by output layer).
    VectorXd sigmoid(const VectorXd &z);                                                        //Sigmoid (activation) function.
    VectorXd Dsigmoid(const VectorXd &z);                                                       //Derivative of sigmoid function.
    void checkSaturation();

    VectorXd feedforward2(VectorXd a);                                                          //Feedforward2.
    void backpropagate3(const VectorXd &trnExp, const VectorXd &y);                             //Backpropagate3.
};

#endif // NETWORK_H_INCLUDED
