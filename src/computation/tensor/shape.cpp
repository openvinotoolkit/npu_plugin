#include "include/mcm/computation/tensor/shape.hpp"

mv::Shape::Shape(std::initializer_list<std::size_t> dims) :
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

mv::Shape::Shape(mv::json::Value& o)
{
    for(unsigned i = 0; i < o.size(); ++i)
        dims_.push_back(constructByteTypeFromJson(o[i]));
}

mv::Shape::Shape()
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

std::size_t& mv::Shape::operator[](int ndim)
{
    if (ndim < 0)
        return dims_.at(dims_.size() + ndim);
    return dims_.at(ndim);
}

const std::size_t& mv::Shape::operator[](int ndim) const
{
    if (ndim < 0)
        return dims_.at(dims_.size() + ndim);
    return dims_.at(ndim);
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

std::string mv::Shape::toString() const
{

    std::string output("(");

    for (std::size_t i = 0; i < dims_.size() - 1; ++i)
    {
        output += std::to_string(dims_[i]);
        output += ", ";
    }

    output += std::to_string(dims_[dims_.size() - 1]);
    output += ")";
    
    return output;

}

mv::json::Value mv::Shape::toJsonValue() const
{

    mv::json::Array arr;

    for (std::size_t i = 0; i < dims_.size(); ++i)
    {
        arr.append(mv::Jsonable::toJsonValue(dims_[i]));
    }

    return mv::json::Value(arr);

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
            {
                return Shape();
            }

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
