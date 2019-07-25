/** INCLUDES                                        */
#include "network.h"

#include <random>
#include <math.h>
#include <stdarg.h>

/** DECLARATIONS                                    */

/** Constructor
*   @param layers Number of layers in network.
*   @param ... Sizes (neuron count) for individual layers.
*/
Network::Network(int layers, ...):mLayers(layers){
    va_list pl;
    va_start(pl, layers);

//Set layer sizes.
    while(layers-- > 0) aSizes.push_back(va_arg(pl, unsigned int));
    va_end(pl);

//Set used neuron count.
    mNeurons=0;
    std::for_each(aSizes.begin(), aSizes.end(), [this](int n){mNeurons+=n;});

//Initalize biases as values from normal distribution.
//There aren't biases for first layer.
    //Create random generator.
    std::default_random_engine generator;
    std::normal_distribution<float> normalDist(0.0, 1.0);
    //Fill bias vectors for each layer.
    for(unsigned int n=1; n<mLayers; n++){
        //Create single bias vector.
        VectorXf biasVec=VectorXf(aSizes[n]);
        //Fill single bias vector by random values from N(0,1).
        biasVec=biasVec.unaryExpr([&generator, &normalDist](float b){return normalDist(generator);});
    }
/*
    //Set size of bias vector.
    biases=Eigen::VectorXf(mNeurons-aSizes[0]);
    //Fill bias vector by random values from N(0, 1).
    for(unsigned int n=0; n<mNeurons-aSizes[0]; n++)
        biases(n)=normalDist(generator);*/

//Initalize layers weights as values from normal distribution.
//There aren't weight matrix for the first layer, so mLayers-1 weight matrices are created.
    for(unsigned int n=0; n<mLayers-1; n++){
        //Cereate new weight matrix.
        Eigen::MatrixXf weightMat(aSizes[n+1], aSizes[n]);
        //Fill weight matrix by random values from N(0, 1).
        for(unsigned int j=0; j<aSizes[n+1]; j++){
            for(unsigned int k=0; k<aSizes[n]; k++){
                weightMat(j, k)=normalDist(generator);
            }
        }
        weightMatrices.push_back(weightMat);
    }

}

/** Sigmoid activation function for layer of neurons. Execute sigmoid function elementwise for each entry in vector z.
*   @param z Neuron layer input values.
*   @return Neuron layer output values.
*/
VectorXf Network::sigmoid(const VectorXf &z){
    //return 1.0/(1.0+exp(-z));
    return z.unaryExpr([](float x){return 1.0f/(1.0f+exp(-x));});
}

/** Derivative of sigmoid function. Execute Dsigmoid function elementwise for each entry in vector z.
*   @param z Input.
*   @return Output.
*/
VectorXf Network::Dsigmoid(const VectorXf &z){
    static VectorXf unitValVector;
    unitValVector.setOnes(z.rows());
    //return sigmoid(z)*(1.0-sigmoid(z));
    return sigmoid(z).cwiseProduct(unitValVector - sigmoid(z));//sigmoid(z).cwiseProduct(sigmoid(-z));
}

/** Feed data stored in input layer trough network and calculate output layer.
*   @param  l Network's input layer.
*   @return Network's output
*/
void Network::feedforward(VectorXf a){
    neuronActivation[0]=a;
    for(unsigned int l=0; l<mLayers-1; l++){
        neuronActivation[l+1]=sigmoid(weightMatrices[l]*neuronActivation[l]+biases[l]);
    }
}

/** Feed data stored in input layer trough network and calculate output layer.
*   @param  l Network's input layer.
*   @return Network's output
*/
VectorXf Network::feedforward2(VectorXf a){
    for(unsigned int l=0; l<mLayers-1; l++){
        a=sigmoid(weightMatrices[l]*a+biases[l]);
    }
    return a;
}

/** Stochastic gradient descent (used learning algorithm).
*   @param miniBatch Set (batch) of randomly picked training examples. Each column of the matrix represent's single tarining example.
*   @param learningRate
*/
void Network::SGD(MatrixXf miniBatch, unsigned int batchSize, float learningRate){
    for(unsigned int n=0; n<batchSize; n++){

    }
}

/**
*
*
*/
void Network::SGD_update(VectorXf trainingExample, float learningRate){

}

/** Calculate new weights & biases based on cost function.
*   @param trnExp Single training example.
*
*/
void Network::backpropagate(VectorXf trnExp){

}

/** Network's cost function.
*   @param y Desired output
*   @return Network's cost value.
*/
float Network::cost(const VectorXf &y){

}

/** Derivative of cost function (respect by output layer).
*   @param y Desired output.
*   @return Error values for neurons in the output layer.
*/
Vectorxf Dcost(const VectorXf &y){

}
