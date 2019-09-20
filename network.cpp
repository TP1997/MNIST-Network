/** INCLUDES                                        */
#include "network.h"

#include <random>
#include <math.h>
#include <stdarg.h>
#include <iostream>
#include <algorithm>

/**---------------------- DECLARATIONS (PUBLIC) ------------------------*/

/** Constructor
*   @param layers, Number of layers in network.
*   @param ..., Sizes (neuron count) for individual layers.
*/
Network::Network(int layers, ...):mLayers(layers){
    va_list pl;
    va_start(pl, layers);

//Set layer sizes.
    aSizes.reserve(mLayers);
    while(layers-- > 0) aSizes.push_back(va_arg(pl, unsigned int));
    va_end(pl);

//Set used neuron count.
    mNeurons=0;
    std::for_each(aSizes.begin(), aSizes.end(), [this](int n){mNeurons+=n;});

//Create random generator.
    std::default_random_engine generator;
    std::normal_distribution<double> normalDist(0.0, 0.3);

//Initalize biases as values from normal distribution.
//There aren't biases for first layer so mLayers-1 bias layers are created.
    generator.seed(rndmSeed());
    biases.reserve(mLayers-1);
    for(unsigned int n=1; n<mLayers; n++){
        //Create single bias vector.
        VectorXd biasVec=VectorXd(aSizes[n]); biasVec.setZero(aSizes[n]);
        //Fill single bias vector by random values from N(0,1).
        //for_each( biasVec.data(), biasVec.data()+biasVec.size(), [&generator, &normalDist](double &b){b=normalDist(generator);} );
        biases.push_back(biasVec);
    }

//Initalize layers weights as values from normal distribution.
//There aren't weight matrix for the first layer so mLayers-1 weight matrices are created.
    generator.seed(rndmSeed());
    weightMatrices.reserve(mLayers-1);
    for(unsigned int n=0; n<mLayers-1; n++){
        //Cereate new weight matrix.
        MatrixXd weightMat(aSizes[n+1], aSizes[n]);
        //Fill weight matrix by random values from N(0, 1).
        for_each( weightMat.data(), weightMat.data()+weightMat.size(), [&generator, &normalDist](double &w){w=normalDist(generator);} );
        weightMatrices.push_back(weightMat);
    }

//Initalize sizes of neuron activations.
    neuronActivation.reserve(mLayers);
    for(unsigned int n=0; n<mLayers; n++){
        neuronActivation.push_back(VectorXd(aSizes[n]));
    }

}
/** Stochastic gradient descent (used learning algorithm).
*   @param trainSet, Set of network inputs & outputs for training session.
*   @param mbSize, Size of minibatch.
*   @param ephocs, Number of individual training sessions.
*   @param eta, Learning rate.
*/
void Network::train(MnistData<VectorXd> &trainSet, unsigned int mbSize, unsigned int epochs, double eta){

    cout << "Training begin" << endl;
    int updates=0;
    for(unsigned int n=0; n<epochs; n++){
        trainSet.shuffle();
        for(unsigned int startIdx=0; startIdx<trainSet.size; startIdx+=mbSize){
            const MatrixXd imgBatch=trainSet.getImgBatch(startIdx, mbSize);
            const MatrixXd lblBatch=trainSet.getLblBatch(startIdx, mbSize);

            SGD_update(imgBatch, lblBatch, eta);
            updates++;
            //cout << startIdx << endl;
        }
        cout << "Epoch " << n+1 << " complete." << endl;
    }
    cout << "Total of " << updates << " minibatch updates." << endl;
}

/** Approximate true GD of cost function by calculating GD against of few randomly selected training examples (minibatch).
*   Update network's weights and biases based on SGD.
*   @param minibatch, Set of randomly selected training examples (network inputs).
*   @param y, Desired output vectors based on minibatch.
*   @param eta, Learning rate.
*/
void Network::SGD_update(const MatrixXd &minibatch, const MatrixXd &y, double eta){

    unsigned int mbSize=minibatch.cols();
    //Create bias error-set for each layer & initalize all subvectors elements to 0.
    vector<VectorXd> biasError(mLayers-1);
    int i=1;
    for_each( biasError.begin(), biasError.end(), [&i, &aSizes=aSizes](VectorXd &v){v.resize(aSizes[i]); v.setZero(aSizes[i++]);} );
    //Create weight error-matrix-set for each layer & initalize all submatrix elements to 0.
    vector<MatrixXd> weightError(mLayers-1);
    i=0;
    for_each( weightError.begin(), weightError.end(), [&i, &aSizes=aSizes](MatrixXd &m){m.resize(aSizes[i+1], aSizes[i]); m.setZero(aSizes[i+1], aSizes[i]); i++;} );

    for(unsigned int n=0; n<mbSize; n++){
    //Backpropagate & get error of single training example.
        vector<VectorXd> error=backpropagate(minibatch.col(n), y.col(n)); //Get nabla_b
    //Add error to total bias errors.
        i=0;
        for_each( biasError.begin(), biasError.end(), [&i, &error](VectorXd &v){v+=error[i++];} );
    //Add error to total weight errors.
        i=0;
        for(MatrixXd &wem : weightError){
            wem+=error[i]*neuronActivation[i].transpose();
            i++;
        }

    }
    //Update biases
    for(unsigned int n=0; n<mLayers-1; n++){
        biases[n] -= eta/double(mbSize) * biasError[n];
    }
    //Update weights.
    for(unsigned int n=0; n<mLayers-1; n++){
        weightMatrices[n] -= eta/double(mbSize) * weightError[n];
    }

}

/** TEMPORARY & DEBUG VERSION OF SGD().
*
*/
void Network::SGD_debug(const vector<VectorXd> &trainSet, const vector<VectorXd> &labelSet, unsigned int mbSize, unsigned int epochs, double eta, const vector<VectorXd> &testSet, const vector<VectorXd> &testLabelSet, vector<double> &result){
    //Resize result vector.
    result.reserve(epochs);

    unsigned int trainSetSize=trainSet.size();
    MatrixXd miniBatch(aSizes[0], mbSize);
    MatrixXd y(10, mbSize);

    for(unsigned int n=0; n<epochs; n++){
        //Create random minibatch & desired output vectors.
        srand(rndmSeed());
        for(unsigned int m=0; m<mbSize; m++){
            unsigned int rndmIdx=rand()%trainSetSize;
            miniBatch.col(m)=trainSet[rndmIdx];
            y.col(m)=labelSet[rndmIdx];
        }
        //Train network with created minibatch.
        //SGD_update(miniBatch, y, eta);
        //Current priori version of ^.
        SGD_updateTemp(miniBatch, y, eta, testSet, testLabelSet, result);
        cout << "Epoch " << n+1 << " complete." << endl;
    }
}

/** TEMPORARY & DEBUG VERSION OF SGD_update().
*
*/
void Network::SGD_updateTemp(const MatrixXd &minibatch, const MatrixXd &y, double eta, const vector<VectorXd> &testSet, const vector<VectorXd> &testLabelSet, vector<double> &result){
    unsigned int mbSize=minibatch.cols();
    //Create bias error-set for each layer & initalize all subvectors elements to 0.
    vector<VectorXd> biasError(mLayers-1);
    int i=1;
    for_each( biasError.begin(), biasError.end(), [&i, &aSizes=aSizes](VectorXd &v){v.resize(aSizes[i]); v.setZero(aSizes[i]); i++;} );
    //Create weight error--matrix-set for each layer & initalize all submatrix elements to 0.
    vector<MatrixXd> weightError(mLayers-1);
    i=0;
    for_each( weightError.begin(), weightError.end(), [&i, &aSizes=aSizes](MatrixXd &m){m.resize(aSizes[i+1], aSizes[i]); m.setZero(aSizes[i+1], aSizes[i]); i++;} );

    for(unsigned int n=0; n<mbSize; n++){
    //Backpropagate & get error of single training example.
        vector<VectorXd> nabla_b=backpropagate(minibatch.col(n), y.col(n)); //Get nabla_b
    //Add error of single example to total bias errors (vectors) of current minibatch.
        i=0;
        for_each( biasError.begin(), biasError.end(), [&i, &nabla_b](VectorXd &v){v+=nabla_b[i++];} );
    //Add error of single example to total weight errors (matrices) of current minibatch.
        i=0;
        for_each( weightError.begin(), weightError.end(), [&i, &nabla_b, this](MatrixXd &m){m+=nabla_b[i]*neuronActivation[i].transpose(); i++;} );
    }

    //Update biases.
    for(unsigned int n=0; n<mLayers-1; n++){
        biases[n] -= eta/double(mbSize) * biasError[n];
    }
    //Update weights.
    for(unsigned int n=0; n<mLayers-1; n++){
        weightMatrices[n] -= eta/double(mbSize)*weightError[n];
    }
    //Get number of recognized images.
    unsigned int recognized=evaluate(testSet, testLabelSet);
    result.push_back(double(recognized)/double(testSet.size())*100.0);
}

/** Test network's performance.
*   @param testSet, Set of network inputs for testing session.
*   @param labelSet, Set of desired outputs based on testSet.
*/
unsigned int Network::evaluate(const vector<VectorXd> &testSet, const vector<VectorXd> &labelSet){
    unsigned int total=testSet.size();
    unsigned int recognized=0;
    for(unsigned int n=0; n<total; n++){
        feedforward(testSet[n]);
        unsigned int netOut, corrOut;
        neuronActivation[mLayers-1].maxCoeff(&netOut);
        labelSet[n].maxCoeff(&corrOut);
        if(netOut==corrOut)
            recognized++;
    }

    return recognized;
}

/**
*
*/
unsigned int Network::evaluate(const MnistData<double> &testSet){
    unsigned int recognized=0;
    for(unsigned int n=0; n<testSet.size; n++){
        feedforward(testSet.getImg(n));
        unsigned int netOut;
        neuronActivation[mLayers-1].maxCoeff(&netOut);
        if(netOut==testSet.getLbl(n))
            recognized++;
    }
    return recognized;
}

/** Recognase single digit.
*   @param input, Input layer data.
*   @return Output
*/
VectorXd Network::recDigit(const VectorXd &input){
    feedforward(input);
    return neuronActivation[mLayers-1];
}


/**--------------------- DECLARATIONS (PRIVATE) ------------------------------*/

/** Feed data stored in input layer trough network and calculate output layer.
*   @param  a, Network's input layer.
*   @return z, Network's z-values (neuron inputs).
*/
vector<VectorXd> Network::feedforward(const VectorXd &a){
    vector<VectorXd> z(mLayers-1);
    neuronActivation[0]=a;
    for(unsigned int l=0; l<mLayers-1; l++){
        z[l]=weightMatrices[l]*neuronActivation[l]+biases[l];
        neuronActivation[l+1]=sigmoid(move(z[l]));
    }

    return z;
}

/** Feedforward data & backpropagate the error.
*   @param trnExp, Single training example.
*   @param y, Desired output vector based on training example.
*   @return error, Set of error values based on neurons activaiton values.
*/
vector<VectorXd> Network::backpropagate(const VectorXd &trnExp, const VectorXd &y){
//Array of neuron errors in specific layers. There isn't error calculated in input layer, ==> layer_1_error = error[0] ...
    vector<VectorXd> error(mLayers-1);
//Feedforward the data.
    vector<VectorXd> z=feedforward(trnExp);
//Backpropagate the error.
    //For the last layer calculate error separately.
    error[mLayers-2]=Dcost(y).cwiseProduct(Dsigmoid(z[mLayers-2]));
    //For remaining layers.
    for(int l=mLayers-3; l>=0; l--){
        error[l]=(weightMatrices[l+1].transpose()*error[l+1]).cwiseProduct(Dsigmoid(z[l]));//?
    }

    return error;
}

/** Derivative of cost function (respect by output layer).
*   @param y Desired output.
*   @return Vector of partial derivatives of const function (respect by output layer activations).
*/
VectorXd Network::Dcost(const VectorXd &y){
    return (neuronActivation[mLayers-1]-y);
}

/** Sigmoid activation function for layer of neurons. Execute sigmoid function elementwise for each entry in vector z.
*   @param z Neuron layer input values.
*   @return Neuron layer output values.
*/
VectorXd Network::sigmoid(const VectorXd &&z){
    //return 1.0/(unitValVector+exp(z));
    return z.unaryExpr([](double x){return 1.0/(1.0+exp(-x));});
}

/** Derivative of sigmoid function. Execute Dsigmoid function elementwise for each entry in vector z.
*   @param z Input.
*   @return Output.
*/
VectorXd Network::Dsigmoid(const VectorXd &z){
    return sigmoid(move(z)).cwiseProduct(sigmoid(-z));
}
