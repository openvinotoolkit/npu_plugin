#include "include/mcm/computation/tensor/tensor.hpp"
#include "include/mcm/computation/tensor/math.hpp"

mv::allocator mv::Tensor::allocator_;
mv::static_vector<mv::dim_type, mv::byte_type, mv::max_ndims> mv::Tensor::subsBuffer_;

const std::function<unsigned(const mv::Shape&, const mv::static_vector<mv::dim_type, mv::byte_type, mv::max_ndims>&)> mv::Tensor::subToIndColumMajor_ = 
    [](const mv::Shape& s, const static_vector<dim_type, byte_type, max_ndims>& sub)
{

    if (s.ndims() == 0)
        throw ShapeError("Cannot compute subscripts for 0-dimensional shape");

    if (sub.length() != s.ndims())
        throw ShapeError("Mismatch between subscript vector and number of dimensions in shape");

    unsigned currentMul = 1;
    unsigned currentResult = 0;

    for (unsigned i = 0; i < sub.length(); ++i)
    {

        if (sub[i] >=  s[i])
            throw ShapeError("Subscript exceeds the dimension");

        currentResult += currentMul * sub[i];
        currentMul *= s[i];

    }

    return currentResult;

};

const std::function<mv::static_vector<mv::dim_type, mv::byte_type, mv::max_ndims>(const mv::Shape&, unsigned)> mv::Tensor::indToSubColumMajor_ = 
    [](const mv::Shape& s, unsigned idx)
{

    if (s.ndims() == 0)
        throw ShapeError("Cannot compute subscripts for 0-dimensional shape");

    static_vector<dim_type, byte_type, max_ndims> sub(s.ndims());
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

};


const std::function<unsigned(const mv::Shape& s, const mv::static_vector<mv::dim_type, mv::byte_type, mv::max_ndims>&)> mv::Tensor::subToIndRowMajor_ =
    [](const mv::Shape& s, const static_vector<dim_type, byte_type, max_ndims>& sub)
{

    if (s.ndims() == 0)
        throw ShapeError("Cannot compute subscripts for 0-dimensional shape");

    if (sub.length() != s.ndims())
        throw ShapeError("Mismatch between subscript vector and number of dimensions in shape");

    unsigned currentMul = 1;
    unsigned currentResult = 0;

    for (int i = sub.length() - 1; i >= 0 ; --i)
    {

        if (sub[i] >=  s[i])
            throw ShapeError("Subscript exceeds the dimension");

        currentResult += currentMul * sub[i];
        currentMul *= s[i];

    }

    return currentResult;

};

const std::function<mv::static_vector<mv::dim_type, mv::byte_type, mv::max_ndims>(const mv::Shape& s, unsigned)> mv::Tensor::indToSubRowMajor_ = 
    [](const mv::Shape& s, unsigned idx)
{

    if (s.ndims() == 0)
        throw ShapeError("Cannot compute subscripts for 0-dimensional shape");

    static_vector<dim_type, byte_type, max_ndims> sub(s.ndims());
    sub[s.ndims() - 1] =  idx % s[s.ndims() - 1];
    int offset = -sub[s.ndims() - 1];
    int scale = s[s.ndims() - 1];
    for (int i = s.ndims() - 2; i >= 0; --i)
    {   
        sub[i] = (idx + offset) / scale % s[i];
        offset -= sub[i] * scale;
        scale *= s[i];
    }

    return sub;

};

const std::function<unsigned(const mv::Shape& s, const mv::static_vector<mv::dim_type, mv::byte_type, mv::max_ndims>&)> mv::Tensor::subToIndPlanar_ = 
    [](const mv::Shape& s, const static_vector<dim_type, byte_type, max_ndims>& sub)
{
    if (s.ndims() == 0)
        throw ShapeError("Cannot compute subscripts for 0-dimensional shape");

    if (sub.length() != s.ndims())
        throw ShapeError("Mismatch between subscript vector and number of dimensions in shape");

    unsigned currentMul = 1;
    unsigned currentResult = 0;

    for (int i = sub.length() - 1; i > 1; --i)
    {

        if (sub[i] >=  s[i])
            throw ShapeError("Subscript exceeds the dimension");

        currentResult += currentMul * sub[i];
        currentMul *= s[i];

    }

    currentResult += currentMul * sub[0];
    currentMul *= s[0];

    if (sub.length() > 1)
        currentResult += currentMul * sub[1];

    return currentResult;

};

const std::function<mv::static_vector<mv::dim_type, mv::byte_type, mv::max_ndims>(const mv::Shape& s, unsigned)>mv::Tensor:: indToSubPlanar_ = 
    [](const mv::Shape& s, unsigned idx)
{

    if (s.ndims() == 0)
        throw ShapeError("Cannot compute subscripts for 0-dimensional shape");

    static_vector<dim_type, byte_type, max_ndims> sub(s.ndims());

    if (s.ndims() == 1)
    {
        sub[0] =  idx % s[0];
        return sub;
    }
    else if (s.ndims() == 2)
    {
        sub[0] = idx % s[0];
        sub[1] = (idx - sub[0]) / s[0] % s[1];
        return sub;
    }
    else
    {
        sub[s.ndims() - 1] =  idx % s[s.ndims() - 1];
        int offset = -sub[s.ndims() - 1];
        int scale = s[s.ndims() - 1];
        for (int i = s.ndims() - 2; i > 1; --i)
        {   
            sub[i] = (idx + offset) / scale % s[i];
            offset -= sub[i] * scale;
            scale *= s[i];
        }
        sub[0] = (idx + offset) / scale % s[0];
        offset -= sub[0] * scale;
        scale *= s[0];
        sub[1] = (idx + offset) / scale % s[1];
    }

    return sub;

};

unsigned mv::Tensor::subToInd(const Shape& shape, const static_vector<dim_type, byte_type, max_ndims>& sub, Order order)
{
    
    auto subToIndFcn = selectSubToInd_(order);
    return subToIndFcn(shape, sub);

}

mv::static_vector<mv::dim_type, mv::byte_type, mv::max_ndims> mv::Tensor::indToSub(const Shape& shape, unsigned idx, Order order)
{

    auto subToIndFcn = selectIndToSub_(order);
    return subToIndFcn(shape, idx);

}

mv::Tensor::Tensor(const string &name, const Shape &shape, DType dType, Order order) :
ComputationElement(name),
errValue(0.0f),
shape_(shape),
populated_(false),
subToIndFcn_(selectSubToInd_(order)),
indToSubFcn_(selectIndToSub_(order))
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
    if(populated_)
        data_ = std::make_shared<dynamic_vector<float_type>>(dynamic_vector<float_type>(shape_.totalSize()));
}

mv::Tensor::Tensor(const string &name, const Shape &shape, DType dType, Order order, const dynamic_vector<float_type>& data) :
Tensor(name, shape, dType, order)
{
    if (populate(data))
        getAttr("populated").setContent<bool>(true);
}

mv::Tensor::Tensor(const Tensor &other) :
ComputationElement(other),
shape_(other.shape_),
populated_(other.populated_),
subToIndFcn_(other.subToIndFcn_),
indToSubFcn_(other.indToSubFcn_)
{
    if (populated_)
        data_ = std::make_shared<dynamic_vector<float_type>>(dynamic_vector<float_type>(*other.data_));
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

mv::Tensor::~Tensor()
{

}

bool mv::Tensor::populate(const dynamic_vector<float_type>& data, Order order)
{

    if (order != Order::Unknown && order != getOrder())
    {
        getAttr("order").setContent<Order>(order);
        selectSubToInd_(order);
        selectIndToSub_(order);
    }

    if (data.size() != getShape().totalSize())
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to populate tensor - mismatch between input array size (" + 
            Printable::toString((unsigned)data.size()) + ") and declared shape (" + getAttr("shape").getContentStr() + ")");
        return false;
    }

    data_ = std::make_shared<dynamic_vector<float>>(data);
    getAttr("populated").setContent<bool>(true);
    populated_ = true;
    return true;

}

bool mv::Tensor::unpopulate()
{
    if (!getAttr("populated").getContent<bool>())
        return false;
    
    data_.reset();
    getAttr("populated").setContent<bool>(false);
    populated_ = false;
    return true;

}

void mv::Tensor::reorder(Order order)
{

    Order oldOrder = getAttr("order").getContent<Order>();

    getAttr("order").setContent<Order>(order);
    selectSubToInd_(order);
    selectIndToSub_(order);

    if (!populated_)
        return;

    auto dataPtr = std::make_shared<dynamic_vector<float>>(data_->size());

    for (unsigned i = 0; i < dataPtr->size(); ++i)
    {

        auto sub = indToSub(shape_, i, oldOrder);
        dataPtr->at(subToInd(shape_, sub, order)) = data_->at(i);

    }

    data_ = dataPtr;

}

bool mv::Tensor::broadcast(const Shape& shape)
{

    if (!isPopulated())
    {
        Tensor::logger().log(Logger::MessageType::MessageError, "Unable to perfom element-wise operation using unpopulated tensor");
        return false;
    }

    Shape s1 = shape_, s2 = shape;

    if (s1.ndims() == 0 || s2.ndims() == 0)
    {
        Tensor::logger().log(Logger::MessageType::MessageError, "Unable to perfom element-wise operation using 0-dimensional tensor");
        return false;
    }

    if (s1 == s2)   
        return true;
    else
    {   

        Shape sO = Shape::broadcast(s1, s2);
        if (sO == Shape())
            return false;

        std::shared_ptr<dynamic_vector<float>> dataPtr = std::make_shared<dynamic_vector<float>>(sO.totalSize());

        if (s1.ndims() > s2.ndims())
        {
            s2 = Shape::augment(s2, s1.ndims());
        }
        else if (s2.ndims() > s1.ndims())
            s1 = Shape::augment(s1, s2.ndims());

        for (unsigned i = 0; i < dataPtr->size(); ++i)
        {
            
            static_vector<dim_type, byte_type, max_ndims> sub = indToSub_(sO, i);

            for (unsigned j = 0; j < sub.length(); ++j)
            {
                if (s1[j] == 1 && sO[j] > 1)
                    sub[j] = 0;
            }

            (*dataPtr)[i] = at(subToInd_(s1, sub));

        }

        shape_ = sO;
        data_ = dataPtr;
        return true; 

    }

    return false;

}

// TODO - Handle the case when tensor got deleted, by the reference is still in use
mv::dynamic_vector<mv::float_type>& mv::Tensor::getData()
{
    if (!isPopulated())
        logger_.log(Logger::MessageType::MessageWarning, "Attempt of restoring data from an unpopulated tensor '" + name_ + "'");
    return *data_.get();
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

bool mv::Tensor::elementWise_(const Tensor& other, const std::function<float(float, float)>& opFunc)
{

    if (!isPopulated() || !other.isPopulated())
    {
        Tensor::logger().log(Logger::MessageType::MessageError, "Unable to perfom element-wise operation using unpopulated tensor");
        return false;
    }

    Shape s1 = shape_, s2 = other.getShape();

    if (s1.ndims() == 0 || s2.ndims() == 0)
    {
        Tensor::logger().log(Logger::MessageType::MessageError, "Unable to perfom element-wise operation using 0-dimensional tensor");
        return false;
    }

    if (s1 == s2)
    {      
        for (unsigned i = 0; i < data_->size(); ++i)
            (*data_)[i] = opFunc(at(i), other(i));
        return true;
    }
    else
    {   

        /*Tensor broadcastOther(other);
        if (!broadcastOther.broadcast(s1))
            return false;
        if (!broadcast(s2))
            return false;

        std::transform(data_->begin(), data_->end(), broadcast.data_->begin(), data_->begin(), opFunc);
        return true;*/

        Shape sO = Shape::broadcast(s1, s2);
        std::shared_ptr<dynamic_vector<float>> dataPtr;

        if (sO == getShape())
        {
            dataPtr = data_;
        }
        else
        {
            dataPtr = std::make_shared<dynamic_vector<float>>(sO.totalSize());
        }

        if (s1.ndims() > s2.ndims())
        {
            s2 = Shape::augment(s2, s1.ndims());
        }
        else if (s2.ndims() > s1.ndims())
            s1 = Shape::augment(s1, s2.ndims());

        for (unsigned i = 0; i < dataPtr->size(); ++i)
        {
            
            static_vector<dim_type, byte_type, max_ndims> subO = indToSub_(sO, i);
            static_vector<dim_type, byte_type, max_ndims> sub1 = subO, sub2 = subO;

            for (unsigned j = 0; j < subO.length(); ++j)
            {
                if (s1[j] == 1 && sO[j] > 1)
                    sub1[j] = 0;
                if (s2[j] == 1 && sO[j] > 1)
                    sub2[j] = 0;
            }

            (*dataPtr)[i] = opFunc(at(subToInd_(s1, sub1)), other.at(subToInd_(s2, sub2)));

        }

        shape_ = sO;
        data_ = dataPtr;
        return true;

    }

    return false;
}

bool mv::Tensor::add(const Tensor& other)
{
    return elementWise_(other, std::plus<float>());
}

bool mv::Tensor::add(float val)
{

    if (!isPopulated())
    {
        Tensor::logger().log(Logger::MessageType::MessageError, "Unable to perfom scalar operation using unpopulated tensor");
        return false;
    }
    for (unsigned i = 0; i < data_->size(); ++i)
        (*data_)[i] += val;
    
    return true;

}

bool mv::Tensor::subtract(const Tensor& other)
{
    return elementWise_(other, std::minus<float>());
}

bool mv::Tensor::subtract(float val)
{

    if (!isPopulated())
    {
        Tensor::logger().log(Logger::MessageType::MessageError, "Unable to perfom scalar operation using unpopulated tensor");
        return false;
    }
    for (unsigned i = 0; i < data_->size(); ++i)
        (*data_)[i] -= val;
    
    return true;

}

bool mv::Tensor::multiply(const Tensor& other)
{
    return elementWise_(other, std::multiplies<float>());
}

bool mv::Tensor::multiply(float val)
{

    if (!isPopulated())
    {
        Tensor::logger().log(Logger::MessageType::MessageError, "Unable to perfom scalar operation using unpopulated tensor");
        return false;
    }
    for (unsigned i = 0; i < data_->size(); ++i)
        (*data_)[i] *= val;
    
    return true;

}

bool mv::Tensor::divide(float val)
{

    if (!isPopulated())
    {
        Tensor::logger().log(Logger::MessageType::MessageError, "Unable to perfom scalar operation using unpopulated tensor");
        return false;
    }
    for (unsigned i = 0; i < data_->size(); ++i)
        (*data_)[i] /= val;
    
    return true;

}

bool mv::Tensor::divide(const Tensor& other)
{
    return elementWise_(other, std::divides<float>());
}

bool mv::Tensor::sqrt()
{
    for (unsigned i = 0; i < data_->size(); ++i)
        (*data_)[i] = std::sqrt((*data_)[i]);

    return true;
}

mv::Logger& mv::Tensor::logger()
{
    return logger_;
}

mv::float_type& mv::Tensor::at(const static_vector<dim_type, byte_type, max_ndims>& sub)
{
    if (!isPopulated())
    {
        logger_.log(Logger::MessageType::MessageError, "Attempt of reading a value from an unpopulated tensor");
        return errValue;
    }

    return (*data_)[subToInd(sub)];
}

const mv::float_type& mv::Tensor::at(const static_vector<dim_type, byte_type, max_ndims>& sub) const
{
    if (!isPopulated())
    {
        logger_.log(Logger::MessageType::MessageError, "Attempt of reading a value from an unpopulated tensor");
        return errValue;
    }

    return (*data_)[subToInd(sub)];
}

mv::float_type& mv::Tensor::at(unsigned idx)
{
    
    if (!isPopulated())
    {
        logger_.log(Logger::MessageType::MessageError, "Attempt of reading a value from an unpopulated tensor");
        return errValue;
    }

    return (*data_)[idx];

}

const mv::float_type& mv::Tensor::at(unsigned idx) const
{

    if (!isPopulated())
    {
        logger_.log(Logger::MessageType::MessageError, "Attempt of reading a value from an unpopulated tensor");
        return errValue;
    }

    return (*data_)[idx];

}

mv::float_type& mv::Tensor::operator()(unsigned idx)
{
    return at(idx);
}

const mv::float_type& mv::Tensor::operator()(unsigned idx) const
{
    return at(idx);
}

mv::float_type& mv::Tensor::operator()(const static_vector<dim_type, byte_type, max_ndims>& sub)
{
    return at(sub);
}

const mv::float_type& mv::Tensor::operator()(const static_vector<dim_type, byte_type, max_ndims>& sub) const
{
    return at(sub);
}
