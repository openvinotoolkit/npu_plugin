#include "include/mcm/computation/tensor/tensor.hpp"
#include "include/mcm/computation/tensor/math.hpp"

mv::allocator mv::Tensor::allocator_;


unsigned mv::Tensor::subToInd(const Shape& shape, const dynamic_vector<unsigned>& sub)
{
    assert(sub.size() == shape.ndims() && "Shape and subs size mismatch");
    assert(sub.size() != 0 && "Cannot compute index for an empty tensor");

    unsigned currentMul = 1;
    unsigned currentResult = 0;

    for (unsigned i = 0; i < sub.size(); ++i)
    {
        assert(sub[i] < shape[i] && "Index exceeds dimension of tensor");
        currentResult += currentMul * sub[i];
        currentMul *= shape[i];
    }

    return currentResult;

}

mv::dynamic_vector<unsigned> mv::Tensor::indToSub(const Shape& s, unsigned idx)
{

    assert(s.ndims() > 0 && "Cannot compute subscripts for 0-dimensional shape");
    dynamic_vector<unsigned> sub(s.ndims());
    sub[0] =  idx % s[0];
    int offset = -sub[0];
    int scale = s[0];
    for (int i = 1; i < s.ndims(); ++i)
    {   
        sub[i] = (idx + offset) / scale % s[i];
        offset -= sub[i] * s[i - 1];
        scale *= s[i];
    }

    return sub;

}

mv::Tensor::Tensor(const string &name, const Shape &shape, DType dType, Order order) :
ComputationElement(name),
errValue(0.0f),
shape_(shape),
populated_(false)
{
    addAttr("dType", AttrType::DTypeType, dType);
    addAttr("order", AttrType::OrderType, order);
    addAttr("shape", AttrType::ShapeType, shape);
    addAttr("populated", AttrType::BoolType, false);
    logger_.log(Logger::MessageType::MessageDebug, "Defined tensor " + toString());
}

mv::Tensor::Tensor(mv::json::Value& v):
ComputationElement(v),
errValue(0.0f),
shape_(attributes_.at("shape").getContent<Shape>()),
populated_(attributes_.at("populated").getContent<bool>())
{
    /*
    if(populated_)
        data_ = constructFloatVectorFromJson(v["data"]);
    */
}

mv::Tensor::Tensor(const string &name, const Shape &shape, DType dType, Order order, const dynamic_vector<float_type>& data) :
Tensor(name, shape, dType, order)
{
    if (populate(data))
        getAttr("populated").setContent<bool>(true);
}

mv::Tensor::Tensor(const Tensor &other) :
ComputationElement(other),
data_(other.data_),
shape_(other.shape_),
populated_(other.populated_)
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

bool mv::Tensor::populate(const dynamic_vector<float_type>& data)
{

    if (data.size() != getShape().totalSize())
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to populate tensor - mismatch between input array size (" + 
            Printable::toString((unsigned)data.size()) + ") and declared shape (" + getAttr("shape").getContentStr() + ")");
        return false;
    }

    data_ = data;
    getAttr("populated").setContent<bool>(true);
    populated_ = true;
    return true;

}

bool mv::Tensor::unpopulate()
{
    if (!getAttr("populated").getContent<bool>())
        return false;
    
    data_.clear();
    getAttr("populated").setContent<bool>(false);
    populated_ = false;
    return true;

}

// TODO - Handle the case when tensor got deleted, by the reference is still in use
mv::dynamic_vector<mv::float_type>& mv::Tensor::getData()
{
    if (!isPopulated())
        logger_.log(Logger::MessageType::MessageWarning, "Attempt of restoring data from an unpopulated tensor '" + name_ + "'");
    return data_;
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

mv::json::Value mv::Tensor::toJsonValue() const
{
    mv::json::Value v = ComputationElement::toJsonValue();
    //TODO - handle populated tensors better
    /*
    if(isPopulated())
        v["data"] = mv::Jsonable::toJsonValue(data_);
    */
    return v;
}


bool mv::Tensor::add(const Tensor& other)
{

    auto newShape = math::elementWise(*this, other, math::elementAdd, data_);
    if (newShape.ndims() > 0)
    {
        getAttr("shape").setContent<Shape>(newShape);
        shape_ = newShape;
        return true;
    }

    return false;

}

bool mv::Tensor::subtract(const Tensor& other)
{
    auto newShape = math::elementWise(*this, other, math::elementSubtract, data_);
    if (newShape.ndims() > 0)
    {
        getAttr("shape").setContent<Shape>(newShape);
        shape_ = newShape;
        return true;
    }

    return false;
}

bool mv::Tensor::mulitply(const Tensor& other)
{
    auto newShape = math::elementWise(*this, other, math::elementMulitply, data_);
    if (newShape.ndims() > 0)
    {
        getAttr("shape").setContent<Shape>(newShape);
        shape_ = newShape;
        return true;
    }

    return false;
}

bool mv::Tensor::divide(const Tensor& other)
{
    auto newShape = math::elementWise(*this, other, math::elementDivide, data_);
    if (newShape.ndims() > 0)
    {
        getAttr("shape").setContent<Shape>(newShape);
        shape_ = newShape;
        return true;
    }

    return false;
}

mv::Logger& mv::Tensor::logger()
{
    return logger_;
}

unsigned mv::Tensor::subToInd(const dynamic_vector<unsigned>& sub) const
{
    return subToInd(getShape(), sub);
}

mv::dynamic_vector<unsigned> mv::Tensor::indToSub(unsigned idx) const
{

    return indToSub(getShape(), idx);

}

mv::float_type& mv::Tensor::at(const dynamic_vector<unsigned>& sub)
{
    if (!isPopulated())
    {
        logger_.log(Logger::MessageType::MessageError, "Attempt of reading a value from an unpopulated tensor");
        return errValue;
    }

    return data_[subToInd(sub)];
}

const mv::float_type& mv::Tensor::at(const dynamic_vector<unsigned>& sub) const
{
    if (!isPopulated())
    {
        logger_.log(Logger::MessageType::MessageError, "Attempt of reading a value from an unpopulated tensor");
        return errValue;
    }

    return data_[subToInd(sub)];
}

mv::float_type& mv::Tensor::at(unsigned idx)
{
    
    if (!isPopulated())
    {
        logger_.log(Logger::MessageType::MessageError, "Attempt of reading a value from an unpopulated tensor");
        return errValue;
    }

    return data_[idx];

}

const mv::float_type& mv::Tensor::at(unsigned idx) const
{

    if (!isPopulated())
    {
        logger_.log(Logger::MessageType::MessageError, "Attempt of reading a value from an unpopulated tensor");
        return errValue;
    }

    return data_[idx];

}

mv::float_type& mv::Tensor::operator()(unsigned idx)
{
    return at(idx);
}

const mv::float_type& mv::Tensor::operator()(unsigned idx) const
{
    return at(idx);
}

mv::float_type& mv::Tensor::operator()(const dynamic_vector<unsigned>& sub)
{
    return at(sub);
}

const mv::float_type& mv::Tensor::operator()(const dynamic_vector<unsigned>& sub) const
{
    return at(sub);
}
