#ifndef COSTFUNC_H_INCLUDED
#define COSTFUNC_H_INCLUDED

#include <Eigen/Dense>

using namespace Eigen;

class CostFunc{
public:
    virtual ~CostFunc();
    virtual double cost(const VectorXd &y)=0;
    virtual VectorXd Dcost(const VectorXd &y)=0;
};

class Quadratic : CostFunc{
public:
    Quadratic();
    ~Quadratic();
    double cost(const VectorXd &y);
    VectorXd Dcost(const VectorXd &y);

};

class CrossEntropy : CostFunc{
public:
    CrossEntropy();
    ~CrossEntropy();
    double cost(const VectorXd &y);
    VectorXd Dcost(const VectorXd &y);

};
#endif // COSTFUNC_H_INCLUDED
