#include "include/fathom/computation/tensor/model_tensor.hpp"

mv::ModelTensor::ModelTensor(const Logger &logger, const string &name, const Shape &shape, DType dType, Order order) : 
Tensor(shape, dType, order),
ComputationElement(logger, name)
{

}

mv::ModelTensor::~ModelTensor()
{
    
}