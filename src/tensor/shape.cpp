#include "include/mcm/tensor/shape.hpp"
#include "math.h"

const std::unordered_map<std::string, std::size_t> mv::Shape::axis_ =
{
    {"W", 0},
    {"H", 1},
    {"C", 2},
    {"N", 3},
    {"K", 3},
    {"B", 4}
};

std::size_t mv::Shape::getAxis(const std::string& axis)
{
    return axis_.at(axis);
}

mv::Shape::Shape(std::initializer_list<std::size_t> dims) :
dims_(dims)
{

}

mv::Shape::Shape(std::vector<std::size_t> dims) :
dims_(dims)
{

}

mv::Shape::Shape(std::size_t ndims) :
dims_(ndims)
{

}

mv::Shape::Shape(const Shape& other) :
dims_(other.dims_)
{

}

mv::Shape::Shape() : Shape({0,0,0,0})
{

}

std::size_t mv::Shape::ndims() const
{
    return dims_.size();
}

std::size_t mv::Shape::totalSize() const
{

    std::size_t result = dims_[0];

    for (std::size_t i = 1; i < dims_.size(); ++i)
        result *= dims_[i];

    return result;

}

bool mv::Shape::isFlat() const
{
    std::size_t totalSize = dims_[0];

    for (std::size_t i = 1; i < dims_.size(); ++i)
        totalSize *= dims_[i];

    return (dims_[mv::IO_BATCH_DIMENSION] == totalSize ||
            dims_[mv::IO_CHANNEL_DIMENSION] == totalSize ||
            dims_[mv::IO_HEIGHT_DIMENSION] == totalSize ||
            dims_[mv::IO_WIDTH_DIMENSION] == totalSize);
}


mv::Shape::operator std::vector<std::size_t>() const
{
    return dims_;
}

mv::Shape::operator std::vector<unsigned>() const
{
    return std::vector<unsigned>(dims_.begin(), dims_.end());
}

std::size_t& mv::Shape::operator[](int ndim)
{

    return const_cast<std::size_t&>(static_cast<const Shape*>(this)->operator[](ndim));

}

const std::size_t& mv::Shape::operator[](int ndim) const
{

    if (ndim >= static_cast<int>(dims_.size()) || static_cast<int>(dims_.size()) + ndim < 0)
        throw ArgumentError(*this, "index subscript", std::to_string(ndim),
            "Exceeds the dimensionality " + std::to_string(ndims()));

    if (ndim < 0)
        return dims_[dims_.size() + ndim];

    return dims_[ndim];

}

const std::size_t& mv::Shape::operator[](const std::string& ndim) const
{
    return this->operator [](getAxis(ndim));
}

mv::Shape& mv::Shape::operator=(const Shape& other)
{
    dims_ = other.dims_;
    return *this;
}

bool mv::Shape::operator==(const Shape& other) const
{
    return dims_ == other.dims_;
}

bool mv::Shape::operator!=(const Shape& other) const
{
    return !operator==(other);
}

mv::Shape mv::Shape::operator/(const Shape& denum) const
{
    if(this->ndims() != denum.ndims())
        throw ArgumentError(*this, " nominator nDims ",std::to_string(this->ndims()),
                "differs from denuminator " + std::to_string(denum.ndims()));

    const mv::Shape& num = *this;
    std::vector<std::size_t> newDims(num.ndims());

    for(unsigned idx = 0; idx < num.ndims(); ++idx)
    {
        newDims[idx] = (unsigned)ceil(((double)num[idx]) / ((double)denum[idx]));
    }

    return mv::Shape(newDims);
}
mv::Shape mv::Shape::operator-(const Shape& subtrahend) const
{
    if(this->ndims() != subtrahend.ndims())
        throw ArgumentError(*this, " minuend nDims ",std::to_string(this->ndims()),
                "differs from subtrahend " + std::to_string(subtrahend.ndims()));

    const mv::Shape& minuend = *this;
    std::vector<std::size_t> newDims(minuend.ndims());

    for(unsigned idx = 0; idx < minuend.ndims(); ++idx)
    {
        //todo:: some raising signal error for negative;
        newDims[idx] = minuend[idx] - subtrahend[idx];
    }

    return mv::Shape(newDims);
}

mv::Shape mv::Shape::operator+(const Shape& addend) const
{
    if(this->ndims() != addend.ndims())
        throw ArgumentError(*this, " addend nDims ",std::to_string(this->ndims()),
                "differs from subtrahend " + std::to_string(addend.ndims()));

    const mv::Shape& augend = *this;
    std::vector<std::size_t> newDims(augend.ndims());

    for(unsigned idx = 0; idx < augend.ndims(); ++idx)
    {
        //todo:: some raising signal error for negative;
        newDims[idx] = augend[idx] + addend[idx];
    }

    return mv::Shape(newDims);
}

mv::Shape mv::Shape::operator *(const Shape& multiplier) const
{
    if(this->ndims() != multiplier.ndims())
        throw ArgumentError(*this, " addend nDims ",std::to_string(this->ndims()),
                "differs from subtrahend " + std::to_string(multiplier.ndims()));

    const mv::Shape& multiplicand = *this;
    std::vector<std::size_t> newDims(multiplicand.ndims());

    for(unsigned idx = 0; idx < multiplicand.ndims(); ++idx)
    {
        newDims[idx] = multiplicand[idx] * multiplier[idx];
    }

    return mv::Shape(newDims);
}

std::string mv::Shape::toString() const
{

    std::string output("{");

    for (std::size_t i = 0; i < dims_.size() - 1; ++i)
    {
        output += std::to_string(dims_[i]);
        output += ", ";
    }

    output += std::to_string(*dims_.rbegin());
    output += "}";

    return output;

}

mv::Shape mv::Shape::broadcast(const Shape& s1, const Shape& s2)
{

    if (s1.ndims() == 0 || s2.ndims() == 0)
        return s1;

    if (s1 == s2)
        return s1;

    const Shape *sM, *sS;

    if (s1.ndims() >= s2.ndims())
    {
        sM = &s1;
        sS = &s2;
    }
    else
    {
        sM = &s2;
        sS = &s1;
    }

    Shape sO(*sM);

    for (std::size_t i = 1; i <= sS->ndims(); ++i)
    {

        if ((*sM)[-i] != (*sS)[-i])
        {

            if ((*sM)[-i] != 1 && (*sS)[-i] != 1)
                throw ShapeError(*sS, "Broadcasting to shape " + sM->toString() + " is impossible for the dimension "
                    + std::to_string(sS->ndims() - 1));

            if ((*sS)[-i] > (*sM)[-i])
                sO[-i] = (*sS)[-i];

        }

    }

    return sO;

}

mv::Shape mv::Shape::augment(const Shape& s, std::size_t ndims)
{

    if (ndims <= s.ndims())
        return s;

    Shape sAug(ndims);

    for (std::size_t i = 0; i < ndims - s.ndims(); ++i)
        sAug[i] = 1;

    for (unsigned i = 0; i < s.ndims(); ++i)
        sAug[i +  ndims - s.ndims()] = s[i];

    return sAug;

}

mv::Shape mv::Shape::augment_major(const Shape& s, std::size_t ndims)
{

    if (ndims <= s.ndims())
        return s;

    Shape sAug(ndims);

    for (std::size_t i = 0; i < s.ndims(); ++i)
        sAug[i] = s[i];

    for (unsigned i = 0; i < ndims - s.ndims(); ++i)
        sAug[i + s.ndims()] = 1;

    return sAug;

}

std::string mv::Shape::getLogID() const
{
    return "Shape:" + toString();
}
