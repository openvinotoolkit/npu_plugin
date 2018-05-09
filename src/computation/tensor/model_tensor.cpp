#include "include/fathom/computation/tensor/model_tensor.hpp"

mv::ModelTensor::ModelTensor(const Logger &logger, const string &name, const Shape &shape, DType dType, Order order) : 
Tensor(shape, dType, order),
ComputationElement(logger, name)
{
    addAttr("dType", AttrType::DTypeType, dType_);
    addAttr("order", AttrType::OrderType, order_);
    addAttr("shape", AttrType::ShapeType, shape_);
}

mv::ModelTensor::ModelTensor(const ModelTensor &other) :
Tensor(other),
ComputationElement(other.logger_, other.name_)
{

}

mv::ModelTensor::~ModelTensor()
{
    
}
