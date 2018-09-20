#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/tensor/math.hpp"

mv::Tensor::Tensor(const std::string &name, const Shape &shape, DType dType, Order order) :
Element(name)
{
    set<Shape>("shape", shape);
    set<Order>("order", order);
    set<DType>("dType", dType);
    set<bool>("populated", false);
}

mv::Tensor::Tensor(const std::string &name, const Shape &shape, DType dType, Order order, const std::vector<double>& data) :
Tensor(name, shape, dType, order)
{
    populate(data, order);
}

mv::Tensor::Tensor(const Tensor &other) :
Element(other)
{
    if (isPopulated())
        data_ = std::make_shared<std::vector<double>>(std::vector<double>(*other.data_));
}

mv::Tensor::~Tensor()
{

}

std::vector<std::size_t> mv::Tensor::indToSub_(const Shape& s, unsigned index) const
{
    return getOrder().indToSub(s, index);
}

unsigned mv::Tensor::subToInd_(const Shape& s, const std::vector<std::size_t>& sub) const
{

    if (hasAttr("master"))
    {
        Shape correctedShape(s);
        std::vector<std::size_t> correctedSub(sub);
        if (hasAttr("leftPadding"))
        {
            auto padding = get<std::vector<std::size_t>>("leftPadding");
            for (std::size_t i = 0; i < padding.size(); ++i)
            {
                correctedShape[i] += padding[i];
                correctedSub[i] += padding[i];
            }
        }
        if (hasAttr("rightPadding"))
        {
            auto padding = get<std::vector<std::size_t>>("rightPadding");
            for (std::size_t i = 0; i < padding.size(); ++i)
                correctedShape[i] += padding[i];
            
        }

        return getOrder().subToInd(correctedShape, correctedSub);

    }

    return getOrder().subToInd(s, sub);

}

void mv::Tensor::populate(const std::vector<double>& data)
{

    if (data.size() != getShape().totalSize())
        throw ArgumentError(*this, "data vector", std::to_string(data.size()), "Unable to populate, data vector size"
            "does not match total size the tensor (" + std::to_string(getShape().totalSize()) + ")");

    data_ = std::make_shared<std::vector<double>>(data);
    set("populated", true);

}

void mv::Tensor::populate(const std::vector<double>& data, Order order)
{
    set<Order>("order",  order);
    populate(data);
}

void mv::Tensor::unpopulate()
{
    if (!isPopulated())
        return;

    data_.reset();
    set<bool>("populated", false);

}

void mv::Tensor::bindData(Tensor& other, const std::vector<std::size_t>& leftPadding, const std::vector<std::size_t> &rightPadding)
{

    if (!other.isPopulated())
        throw ArgumentError(*this, "InputTensor::populated", "false", "Unable to bind data from an unpopulated tensor " + other.getName());

    if (leftPadding.size() != other.getShape().ndims() && !leftPadding.empty())
        throw ArgumentError(*this, "leftPadding::size", std::to_string(leftPadding.size()), "Invalid dimension of the left padding vector,"
            " must be equal to the dimensionality of the master tensor, which is " + std::to_string(other.getShape().ndims()));

    if (rightPadding.size() != other.getShape().ndims() && !rightPadding.empty())
        throw ArgumentError(*this, "rightPadding::size", std::to_string(rightPadding.size()), "Invalid dimension of the right padding vector,"
            " must be equal to the dimensionality of the master tensor, which is " + std::to_string(getShape().ndims()));

    data_ = other.data_;
    set<bool>("populated", true);
    if (!leftPadding.empty())
        set<std::vector<std::size_t>>("leftPadding", leftPadding);
    if (!rightPadding.empty())
        set<std::vector<std::size_t>>("rightPadding", rightPadding);

    Shape newShape(other.getShape());
    
    for (std::size_t i = 0; i < newShape.ndims(); ++i)
        newShape[i] -= leftPadding[i] + rightPadding[i];

    set<Shape>("shape", newShape);
    set<Order>("order", other.getOrder());
    set<DType>("dType", other.getDType());
    set<std::string>("master", other.getName());
    other.set<std::string>("slave", getName());

}

void mv::Tensor::setOrder(Order order)
{

    if (!isPopulated())
    {
        set<Order>("order", order);
        return;
    }
    
    if (hasAttr("master") || hasAttr("slave"))
        throw ValueError(*this, "Unable to reorder bound tensor");

    auto dataPtr = std::make_shared<std::vector<double>>(data_->size());

    for (unsigned i = 0; i < dataPtr->size(); ++i)
    {

        auto sub = getOrder().indToSub(getShape(), i);
        dataPtr->at(order.subToInd(getShape(), sub)) = data_->at(i);

    }

    set<Order>("order", order);
    data_ = dataPtr;

}

void mv::Tensor::broadcast(const Shape& shape)
{

    if (!isPopulated())
        throw ValueError(*this, "Broadcastring of an unpopulated tensor is undefined");

    if (hasAttr("master") || hasAttr("slave"))
        throw ValueError(*this, "Unable to broadcast a bound tensor"); 

    Shape s1 = getShape(), s2 = shape;

    if (s1.ndims() == 0 || s2.ndims() == 0)
        throw ValueError(*this, "Unable to perfom element-wise operation using 0-dimensional tensor");

    if (s1 != s2)
    {

        // Will throw on error
        Shape sO = Shape::broadcast(s1, s2);

        std::shared_ptr<std::vector<double>> dataPtr = std::make_shared<std::vector<double>>(sO.totalSize());

        if (s1.ndims() > s2.ndims())
        {
            s2 = Shape::augment(s2, s1.ndims());
        }
        else if (s2.ndims() > s1.ndims())
            s1 = Shape::augment(s1, s2.ndims());

        for (unsigned i = 0; i < dataPtr->size(); ++i)
        {

            std::vector<std::size_t> sub = indToSub_(sO, i);

            for (unsigned j = 0; j < sub.size(); ++j)
            {
                if (s1[j] == 1 && sO[j] > 1)
                    sub[j] = 0;
            }

            (*dataPtr)[i] = at(subToInd_(s1, sub));

        }

        set<Shape>("shape", sO);
        data_ = dataPtr;

    }

}

// TODO - Handle the case when tensor got deleted, by the reference is still in use
std::vector<double>& mv::Tensor::getData()
{
    if (!isPopulated())
        throw ValueError(*this, "Attempt of restoring data from an unpopulated tensor");
    return *data_.get();
}

mv::DType mv::Tensor::getDType() const
{
    return get<DType>("dType");
}

mv::Order mv::Tensor::getOrder() const
{
    return get<Order>("order");
}

std::string mv::Tensor::toString() const
{
    return getLogID() + Element::attrsToString_();
}

void mv::Tensor::elementWise_(const Tensor& other, const std::function<double(double, double)>& opFunc)
{

    if (!isPopulated() || !other.isPopulated())
        throw ValueError(*this, "Unable to perfom element-wise operation using unpopulated tensor");

    Shape s1 = getShape(), s2 = other.getShape();

    if (s1.ndims() == 0 || s2.ndims() == 0)
        throw ValueError(*this, "Unable to perfom element-wise operation using 0-dimensional tensor");

    if (s1 == s2)
    {
        for (unsigned i = 0; i < data_->size(); ++i)
            (*data_)[i] = opFunc(at(i), other(i));
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
        std::shared_ptr<std::vector<double>> dataPtr;

        if (sO == getShape())
        {
            dataPtr = data_;
        }
        else
        {
            dataPtr = std::make_shared<std::vector<double>>(sO.totalSize());
        }

        if (s1.ndims() > s2.ndims())
        {
            s2 = Shape::augment(s2, s1.ndims());
        }
        else if (s2.ndims() > s1.ndims())
            s1 = Shape::augment(s1, s2.ndims());

        for (unsigned i = 0; i < dataPtr->size(); ++i)
        {

            std::vector<std::size_t> subO = indToSub_(sO, i);
            std::vector<std::size_t> sub1 = subO, sub2 = subO;

            for (unsigned j = 0; j < subO.size(); ++j)
            {
                if (s1[j] == 1 && sO[j] > 1)
                    sub1[j] = 0;
                if (s2[j] == 1 && sO[j] > 1)
                    sub2[j] = 0;
            }

            (*dataPtr)[i] = opFunc(at(subToInd_(s1, sub1)), other.at(subToInd_(s2, sub2)));

        }

        set<Shape>("shape", sO);
        data_ = dataPtr;

    }

}

void mv::Tensor::add(const Tensor& other)
{
    elementWise_(other, std::plus<double>());
}

void mv::Tensor::add(double val)
{

    if (!isPopulated())
        throw ValueError(*this, "Unable to perfom scalar addition operation for an unpopulated tensor");
    
    for (unsigned i = 0; i < data_->size(); ++i)
        (*data_)[i] += val;

}

void mv::Tensor::subtract(const Tensor& other)
{
    elementWise_(other, std::minus<double>());
}

void mv::Tensor::subtract(double val)
{

    if (!isPopulated())
        throw ValueError(*this, "Unable to perfom scalar subtraction operation for an unpopulated tensor");

    for (unsigned i = 0; i < data_->size(); ++i)
        (*data_)[i] -= val;

}

void mv::Tensor::multiply(const Tensor& other)
{
    elementWise_(other, std::multiplies<double>());
}

void mv::Tensor::multiply(double val)
{

    if (!isPopulated())
        throw ValueError(*this, "Unable to perfom scalar multiplication operation for an unpopulated tensor");

    for (unsigned i = 0; i < data_->size(); ++i)
        (*data_)[i] *= val;

}

void mv::Tensor::divide(double val)
{

    if (!isPopulated())
        throw ValueError(*this, "Unable to perfom scalar division operation for an unpopulated tensor");

    for (unsigned i = 0; i < data_->size(); ++i)
        (*data_)[i] /= val;

}

void mv::Tensor::divide(const Tensor& other)
{
    elementWise_(other, std::divides<double>());
}

void mv::Tensor::sqrt()
{

    if (!isPopulated())
        throw ValueError(*this, "Unable to perfom scalar square root operation for an unpopulated tensor");

    for (unsigned i = 0; i < data_->size(); ++i)
        (*data_)[i] = std::sqrt((*data_)[i]);

}

double& mv::Tensor::at(const std::vector<std::size_t>& sub)
{
    if (!isPopulated())
        throw ValueError(*this, "Unable to access the data value for an unpopulated tensor");

    return (*data_)[subToInd(sub)];
}

const double& mv::Tensor::at(const std::vector<std::size_t>& sub) const
{
    if (!isPopulated())
        throw ValueError(*this, "Unable to access the data value for an unpopulated tensor");

    return (*data_)[subToInd(sub)];
}

double& mv::Tensor::at(std::size_t idx)
{

    return const_cast<double&>(static_cast<const Tensor*>(this)->at(idx));

}

const double& mv::Tensor::at(std::size_t idx) const
{

    if (!isPopulated())
        throw ValueError(*this, "Unable to access the data value for an unpopulated tensor");

    if (idx > data_->size())
        throw IndexError(*this, idx, "Exceeds the total lenght of data vector");

    if (hasAttr("master"))
    {
        std::vector<std::size_t> sub = indToSub(idx);
        return (*data_)[subToInd(sub)];

    }

    return (*data_)[idx];

}

double& mv::Tensor::operator()(std::size_t idx)
{
    return at(idx);
}

const double& mv::Tensor::operator()(std::size_t idx) const
{
    return at(idx);
}

double& mv::Tensor::operator()(const std::vector<std::size_t>& sub)
{
    return at(sub);
}

const double& mv::Tensor::operator()(const std::vector<std::size_t>& sub) const
{
    return at(sub);
}

std::string mv::Tensor::getLogID() const
{
    return "Tensor " + getName();
}