#include "include/fathom/computation/tensor/model_tensor.hpp"

mv::ModelTensor::ModelTensor(const Logger &logger, const string &name, const Shape &shape, DType dType, Order order) : 
Tensor(shape, dType, order),
ComputationElement(logger, name)
{
    logger_.log(Logger::MessageType::MessageDebug, "Defined model tensor " + toString());
    addAttr("dType", AttrType::DTypeType, dType_);
    addAttr("order", AttrType::OrderType, order_);
    addAttr("shape", AttrType::ShapeType, shape_);
}

mv::ModelTensor::ModelTensor(const ModelTensor &other) :
ModelTensor(other.logger_, other.name_, other.shape_, other.dType_, other.order_)
{

}

mv::ModelTensor::ModelTensor(const Logger &logger, const string &name, const ConstantTensor &other) :
ModelTensor(logger, name, other.getShape(), other.getDType(), other.getOrder())
{
    
}

mv::ModelTensor::ModelTensor(const Logger &logger) : 
ComputationElement(logger, "unknown_tensor")
{
    logger_.log(Logger::MessageType::MessageWarning, "Defined unknown model tensor");
}

mv::ModelTensor::~ModelTensor()
{
    
}

mv::string mv::ModelTensor::toString() const
{
    return "'" + name_ + "' " + ComputationElement::toString();
}

/*mv::ModelTensor& mv::ModelTensor::operator=(const ModelTensor &other)
{
    shape_ = other.shape_;
    dType_ = other.dType_;
    order_ = other.order_;
    name_ = other.name_;
    id_ = other.id_;
    attributes_ = other.attributes_;
    return *this;
}*/
