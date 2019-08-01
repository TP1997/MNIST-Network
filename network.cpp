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
    biases.resize(mLayers-1);
    for(unsigned int n=1; n<mLayers; n++){
        //Create single bias vector.
        VectorXf biasVec=VectorXf(aSizes[n]);
        //Fill single bias vector by random values from N(0,1).
        biasVec=biasVec.unaryExpr([&generator, &normalDist](float b){return normalDist(generator);});
        //Put bias vector into array.
        biases.push_back(biasVec);
    }
/*
    //Set size of bias vector.
    biases=Eigen::VectorXf(mNeurons-aSizes[0]);
    //Fill bias vector by random values from N(0, 1).
    for(unsigned int n=0; n<mNeurons-aSizes[0]; n++)
        biases(n)=normalDist(generator);*/

//Initalize layers weights as values from normal distribution.
//There aren't weight matrix for the first layer, so mLayers-1 weight matrices are created.
    weightMatrices.resize(mLayers-1);
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

/** Feed data stored in input layer trough network and calculate output layer.
*   @param  l, Network's input layer.
*   @return z, Network's z-values (neuron inputs).
*/
vector<VectorXf> Network::feedforward(VectorXf a){
    vector<VectorXf> z(mLayers-1);
    neuronActivation[0]=a;
    for(unsigned int l=0; l<mLayers-1; l++){
        z[l]=weightMatrices[l]*neuronActivation[l]+biases[l];
        neuronActivation[l+1]=sigmoid(z[l]);
    }

    return z;
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
*   @param batchSize
*   @param leraningRate
*/
void Network::SGD(MatrixXf miniBatch, unsigned int batchSize, float learningRate){
    for(unsigned int n=0; n<batchSize; n++){

    }
}

/** Approximate true GD of cost function by calculating GD against of few randomly selected training examples (minibatch).
*   Update network's weights and biases based on SGD.
*   @param minibatch, Set of randomly selected training examples.
*   @param corrOutput, Desired output vectors based on minibatch.
*   @param eta, Learning rate.
*/
void Network::SGD_update(const MatrixXf &minibatch, const MatrixXf &corrOutput, float eta){
    //Create error-set for each layer & initalize all subvectors elements to 0.
    vector<VectorXf> totError(mLayers-1);
    int i=1;
    for_each( totError.begin(), totError.end(), [&i, &aSizes=aSizes](VectorXf &v){v.resize(aSizes[i++]);} );

    for(unsigned int n=0; n<minibatch.cols(); n++){
    //Backpropagate & get error of single training example.
        vector<VectorXf> error=backpropagate2(minibatch.col(n), corrOutput.col(n));
    //Add error to total errors of current minibatch.
        i=0;
        for_each( totError.begin(), totError.end(), [&i, &error](VectorXf &v){v+=error[i++];} );
    //Continue...
        //Layer error values in bias-vector.
        //backpropagate3(minibatch.col(n), corrOutput.col(n));
    }
}

/** Feedforward data & backpropagate the error.
*   @param trnExp, Single training example.
*   @param y, Desired output vector based on training example.
*   @return error, Set of errors for weights & biases.
*/
void Network::backpropagate(const VectorXf &trnExp, const VectorXf &y){

}
/** Feedforward data & backpropagate the error.
*   @param trnExp, Single training example.
*   @param y, Desired output vector based on training example.
*   @return error, Set of error values based on neurons activaiton values.
*/
vector<VectorXf> Network::backpropagate2(const VectorXf &trnExp, const VectorXf &y){
//Array of neuron errors in specific layers. There isn't error calculated in input layer, ==> layer_1_error = error[0] ...
    vector<VectorXf> error(mLayers-1);
    //Feedforward the data.
    vector<VectorXf> z=feedforward(trnExp);
//Check cost value.
    //Backpropagate the error.
    //For the last layer calculate error separately.
    error[mLayers-2]=Dcost(y).cwiseProduct(Dsigmoid(z[mLayers-2]));
    //For remaining layers.
    for(unsigned int l=mLayers-3; l>=0; l--){
        error[l]=(weightMatrices[l+1].transpose()*error[l+2]).cwiseProduct(Dsigmoid(z[l]));
    }

    return error;
}
/** Backpropagate 3.
*
*
*/
void Network::backpropagate3(const VectorXf &trnExp, const VectorXf &y){
//Feedforward the data & get neurons input values.
    vector<VectorXf> z=feedforward(trnExp);
//Backpropagate the error.
    //For the last layer calculate error separately.
    biases[mLayers-2]=Dcost(y).cwiseProduct(Dsigmoid(z[mLayers-2]));
    //For remaining layers.
    for(unsigned int l=mLayers-3; l>=0; l--){
        biases[l]=(weightMatrices[l+1].transpose()*biases[l+2]).cwiseProduct(Dsigmoid(z[l]));
    }
}
/** Network's cost function.
*   @param y, Desired output
*   @return costVal, Network's cost value.
*/
float Network::cost(const VectorXf &y){
    float costVal;


    return costVal;
}

/** Derivative of cost function (respect by output layer).
*   @param y Desired output.
*   @return Vector of partial derivatives of const function (respect by output layer activations).
*/
VectorXf Network::Dcost(const VectorXf &y){
    return (neuronActivation[mLayers-1]-y);
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
