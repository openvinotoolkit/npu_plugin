#include "include/mcm/computation/tensor/tensor.hpp"
#include "include/mcm/computation/tensor/math.hpp"

mv::allocator mv::Tensor::allocator_;

mv::Tensor::Tensor(const string &name, const Shape &shape, DType dType, Order order) :
ComputationElement(name),
errValue(0.0f)
{
    addAttr("dType", AttrType::DTypeType, dType);
    addAttr("order", AttrType::OrderType, order);
    addAttr("shape", AttrType::ShapeType, shape);
    addAttr("populated", AttrType::BoolType, false);
    logger_.log(Logger::MessageType::MessageDebug, "Defined tensor " + toString());
}

mv::Tensor::Tensor(const string &name, const Shape &shape, DType dType, Order order, allocator::owner_ptr<dynamic_vector<float_type>> data) :
Tensor(name, shape, dType, order)
{   
    if (data->size() == shape.totalSize())
    {
        data_ = data;
        getAttr("populated").setContent<bool>(true);
    }
    else
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to populate tensor - mismatch between input array size (" + 
            Printable::toString((unsigned)data->size()) + ") and declared shape " + Printable::toString(shape));
    }

}

mv::Tensor::Tensor(const string &name, const Shape &shape, DType dType, Order order, dynamic_vector<float_type>& data) :
Tensor(name, shape, dType, order)
{
    if (populate(data))
        getAttr("populated").setContent<bool>(true);
}

mv::Tensor::Tensor(const Tensor &other) :
ComputationElement(other),
data_(other.data_)
{
    logger_.log(Logger::MessageType::MessageDebug, "Copied tensor " + toString());
}

mv::Tensor::Tensor() :
ComputationElement("unknown_tensor")
{
    logger_.log(Logger::MessageType::MessageWarning, "Defined unknown tensor");
    addAttr("dType", AttrType::DTypeType, DType::Unknown);
    addAttr("order", AttrType::OrderType, Order::Unknown);
    addAttr("shape", AttrType::ShapeType, Shape());
    addAttr("populated", AttrType::BoolType, false);
}

bool mv::Tensor::populate(dynamic_vector<float_type>& data)
{

    if (data.size() != getShape().totalSize())
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to populate tensor - mismatch between input array size (" + 
            Printable::toString((unsigned)data.size()) + ") and declared shape (" + getAttr("shape").getContentStr() + ")");
        return false;
    }

    data_ = allocator_.make_owner<dynamic_vector<float_type>>(data);

    if (data_)
    {
        getAttr("populated").setContent<bool>(true);
        return true;
    }
    
    logger_.log(Logger::MessageType::MessageError, "Unable to populate tensor - allocation failed");
    return false;

}

bool mv::Tensor::unpopulate()
{
    if (!getAttr("populated").getContent<bool>())
        return false;
    
    data_->clear();
    getAttr("populated").setContent<bool>(false);
    return true;

}

bool mv::Tensor::isPopulated() const
{
    return getAttr("populated").getContent<bool>();
}

// TODO - Handle the case when tensor got deleted, by the reference is still in use
mv::dynamic_vector<mv::float_type>& mv::Tensor::getData()
{
    if (!isPopulated())
        logger_.log(Logger::MessageType::MessageWarning, "Attempt of restoring data from an unpopulated tensor '" + name_ + "'");
    return *data_;
}

mv::Shape mv::Tensor::getShape() const
{
    return getAttr("shape").getContent<Shape>();
}

mv::DType mv::Tensor::getDType() const
{
    return getAttr("dType").getContent<DType>();
}

mv::Order mv::Tensor::getOrder() const
{
    return getAttr("order").getContent<Order>();
}

mv::string mv::Tensor::toString() const
{
    return "tensor '" + name_ + "' " + ComputationElement::toString();
}

bool mv::Tensor::add(const Tensor& other)
{

    auto newShape = math::elementWise(*this, other, math::elementAdd, *data_);
    if (newShape.ndims() > 0)
    {
        getAttr("shape").setContent<Shape>(newShape);
        return true;
    }

    return false;

}

bool mv::Tensor::subtract(const Tensor& other)
{
    auto newShape = math::elementWise(*this, other, math::elementSubtract, *data_);
    if (newShape.ndims() > 0)
    {
        getAttr("shape").setContent<Shape>(newShape);
        return true;
    }

    return false;
}

bool mv::Tensor::mulitply(const Tensor& other)
{
    auto newShape = math::elementWise(*this, other, math::elementMulitply, *data_);
    if (newShape.ndims() > 0)
    {
        getAttr("shape").setContent<Shape>(newShape);
        return true;
    }

    return false;
}

bool mv::Tensor::divide(const Tensor& other)
{
    auto newShape = math::elementWise(*this, other, math::elementDivide, *data_);
    if (newShape.ndims() > 0)
    {
        getAttr("shape").setContent<Shape>(newShape);
        return true;
    }

    return false;
}

mv::Logger& mv::Tensor::logger()
{
    return logger_;
}