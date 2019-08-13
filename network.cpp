/** INCLUDES                                        */
#include "network.h"

#include <random>
#include <math.h>
#include <stdarg.h>
#include <iostream>

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
        VectorXd biasVec=VectorXd(aSizes[n]);
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

/** Destructor
*
*/
Network::~Network(){

}

/** Stochastic gradient descent (used learning algorithm).
*   @param trainSet, Set of network inputs for training session.
*   @param labelSet, Set of desired outputs based on trainSet.
*   @param mbSize, Size of minibatch.
*   @param ephocs, Number of individual training sessions.
*   @param eta, Learning rate.
*/
void Network::SGD(const vector<VectorXd> &trainSet, const vector<VectorXd> &labelSet, unsigned int mbSize, unsigned int epochs, double eta){
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
        SGD_update(miniBatch, y, eta);
        cout << "Epoch " << n+1 << " complete." << endl;
    }
}

/** Approximate true GD of cost function by calculating GD against of few randomly selected training examples (minibatch).
*   Update network's weights and biases based on SGD.
*   @param minibatch, Set of randomly selected training examples (network inputs).
*   @param corrOutput, Desired output vectors based on minibatch.
*   @param eta, Learning rate.
*/
void Network::SGD_update(const MatrixXd &minibatch, const MatrixXd &y, double eta){

    unsigned int mbSize=minibatch.cols();
    //Create error-set for each layer & initalize all subvectors elements to 0.
    vector<VectorXd> totError(mLayers-1);
    int i=1;
    for_each( totError.begin(), totError.end(), [&i, &aSizes=aSizes](VectorXd &v){v.resize(aSizes[i]); v.setZero(aSizes[i]); i++;} );

    for(unsigned int n=0; n<mbSize; n++){
    //Backpropagate & get error of single training example.
        vector<VectorXd> error=backpropagate2(minibatch.col(n), y.col(n)); //Get nabla_b
    //Add error of single example to total errors of current minibatch.
        i=0;
        for_each( totError.begin(), totError.end(), [&i, &error](VectorXd &v){v+=error[i++];} );

    }
    //Continue...Update weights & biases.
    //Update biases (by averaging over total error).
    for(unsigned int n=0; n<mLayers-1; n++){
        biases[n] -= double(eta/double(mbSize)) * totError[n];
    }
    //Update weights.

    for(unsigned int n=0; n<mLayers-1; n++){
        MatrixXd errMat = totError[n]*neuronActivation[n].transpose();
        errMat *= double(eta/double(mbSize));
        weightMatrices[n] -= errMat;
    }
    //checkSaturation();
    //cout << m1-m2 << endl;
 /*   for(int n=mLayers-2; n>=1; n--){
        MatrixXd errMat = totError[n]*neuronActivation[n-1].transpose();
        errMat *= double(eta/double(mbSize));
        weightMatrices[n] -= errMat;
    }*/

}

/*###############################################################
*###################3###########################################3
*/
void Network::SGD_update2(const MatrixXd &minibatch, const MatrixXd &y, double eta){
    unsigned int mbSize=minibatch.cols();
    //Create error-set for each layer & initalize all subvectors elements to 0.
    vector<VectorXd> d_Nabla_b(mLayers-1);
    int i=1;
    for_each( d_Nabla_b.begin(), d_Nabla_b.end(), [&i, &aSizes=aSizes](VectorXd &v){v.resize(aSizes[i]); v.setZero(aSizes[i]); i++;} );

    vector<MatrixXd> d_Nabla_w(mLayers-1);
    i=0;
    for_each( d_Nabla_w.begin(), d_Nabla_w.end(), [&i, this](MatrixXd &m){m.resize(weightMatrices[i].rows(), weightMatrices[i].cols()); i++;} );

    for(unsigned int n=0; n<mbSize; n++){
    //Backpropagate & get error of single training example.
        vector<VectorXd> nabla_b=backpropagate2(minibatch.col(n), y.col(n)); //Get nabla_b
        vector<MatrixXd> nabla_w(mLayers-1);
        for(int m=0; m<mLayers-1; m++){
            nabla_w[m].resize(aSizes[m+1], aSizes[m]);
            for(int k=0; k<aSizes[m]; k++){
                for(int j=0; j<aSizes[m+1]; j++){
                    nabla_w[m](j,k)=neuronActivation[m](k)*nabla_b[m](j);
                }
            }
            //nabla_w[m]=nabla_b[m]*neuronActivation[m].transpose();
        }
    //Add error of single example to total errors of current minibatch.
        i=0;
        for_each( d_Nabla_b.begin(), d_Nabla_b.end(), [&i, &nabla_b](VectorXd &v){v+=nabla_b[i++];} );
        i=0;
        for_each( d_Nabla_w.begin(), d_Nabla_w.end(), [&i, &nabla_w](MatrixXd &m){m+=nabla_w[i++];} );

    }
    //Continue...Update weights & biases.
    //Update biases (by averaging over total error).
    for(unsigned int n=0; n<mLayers-1; n++){
        biases[n] -= double(eta/double(mbSize)) * d_Nabla_b[n];
    }
    double fac=double(eta/double(mbSize));
    for(unsigned int n=0; n<mLayers-1; n++){
        d_Nabla_w[n]*=fac;
        weightMatrices[n]-=d_Nabla_w[n];
    }
}

/** Test network's performance.
*   @param testSet, Set of network inputs for testing session.
*   @param labelSet, Set of desired outputs based on testSet.
*/
void Network::evaluate(const vector<VectorXd> &testSet, const vector<VectorXd> &labelSet){
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
    cout << recognized << "/" << total << " images recognized correctly" << endl;
}

/** Get the output vector of the network
*   @return Network's output vector.
*/
VectorXd Network::getOutput(){
    return neuronActivation[mLayers-1];
}

/** Recognase single digit.
*   @param input, Input layer data.
*   @return
*/
VectorXd Network::recDigit(const VectorXd &input){
    feedforward(input);
    return getOutput();
}


/**--------------------- DECLARATIONS (PRIVATE) ------------------------------*/

/** Feed data stored in input layer trough network and calculate output layer.
*   @param  a, Network's input layer.
*   @return z, Network's z-values (neuron inputs).
*/
vector<VectorXd> Network::feedforward(VectorXd a){
    vector<VectorXd> z(mLayers-1);
    neuronActivation[0]=a;
    for(unsigned int l=0; l<mLayers-1; l++){
        z[l]=weightMatrices[l]*neuronActivation[l]+biases[l];
        neuronActivation[l+1]=sigmoid(z[l]);
    }

    return z;
}

/** Feedforward data & backpropagate the error.
*   @param trnExp, Single training example.
*   @param y, Desired output vector based on training example.
*   @return error, Set of error values based on neurons activaiton values.
*/
vector<VectorXd> Network::backpropagate2(const VectorXd &trnExp, const VectorXd &y){
//Array of neuron errors in specific layers. There isn't error calculated in input layer, ==> layer_1_error = error[0] ...
    vector<VectorXd> error(mLayers-1);
    //Feedforward the data.
    vector<VectorXd> z=feedforward(trnExp);
//Check cost value.
    //Backpropagate the error.
    //For the last layer calculate error separately.
    error[mLayers-2]=Dcost(y).cwiseProduct(Dsigmoid(z[mLayers-2]));
    //For remaining layers.
    for(int l=mLayers-3; l>=0; l--){
        error[l]=(weightMatrices[l+1].transpose()*error[l+1]).cwiseProduct(Dsigmoid(z[l]));//!!!
    }

    return error; //return nabla_b;
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
VectorXd Network::sigmoid(const VectorXd &z){
    //return 1.0/(1.0+exp(-z));
    return z.unaryExpr([](double x){return 1.0/(1.0+exp(-x));});
}

/** Derivative of sigmoid function. Execute Dsigmoid function elementwise for each entry in vector z.
*   @param z Input.
*   @return Output.
*/
VectorXd Network::Dsigmoid(const VectorXd &z){
    VectorXd unitValVector;
    unitValVector.setOnes(z.rows());
    //return sigmoid(z)*(1.0-sigmoid(z));
    return sigmoid(z).cwiseProduct(unitValVector - sigmoid(z));//sigmoid(z).cwiseProduct(sigmoid(-z));
}

/**
*
*/
void Network::checkSaturation(){
    double uRange=0.95;
    double dRange=0.05;
    int n=0;
    for(VectorXd &v : neuronActivation){
        unsigned int saturated=0;
        for_each(v.data(), v.data()+v.size(), [&saturated, dRange, uRange](double d){if(d<dRange || d>uRange) saturated++;} );
        cout << "Layer " << ++n << " has " << saturated << "/" << v.size() <<  " saturated neurons." << endl;
    }
}

/**-------------------- DECLARATIONS (TRASH) ---------------------------*/

/** Feed data stored in input layer trough network and calculate output layer.
*   @param  l Network's input layer.
*   @return Network's output
*/
VectorXd Network::feedforward2(VectorXd a){
    for(unsigned int l=0; l<mLayers-1; l++){
        a=sigmoid(weightMatrices[l]*a+biases[l]);
    }
    return a;
}

/** Backpropagate 3.
*
*
*/
void Network::backpropagate3(const VectorXd &trnExp, const VectorXd &y){
//Feedforward the data & get neurons input values.
    vector<VectorXd> z=feedforward(trnExp);
//Backpropagate the error.
    //For the last layer calculate error separately.
    biases[mLayers-2]=Dcost(y).cwiseProduct(Dsigmoid(z[mLayers-2]));
    //For remaining layers.
    for(unsigned int l=mLayers-3; l>=0; l--){
        biases[l]=(weightMatrices[l+1].transpose()*biases[l+2]).cwiseProduct(Dsigmoid(z[l]));
    }
}
