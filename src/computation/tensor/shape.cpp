#include "include/mcm/computation/tensor/shape.hpp"

void mv::Shape::addDim(dim_type newDim)
{
    dims_.push_back(newDim);
}

mv::Shape::Shape(const Shape& other) :
dims_(other.dims_)
{

}

mv::Shape::Shape(byte_type n)
{
    for (unsigned i = 0; i < n; ++i)
        addDim(0);
}

mv::Shape::Shape()
{
    
}

mv::byte_type mv::Shape::ndims() const
{
    return dims_.length();
}

mv::dim_type& mv::Shape::dim(byte_type ndim)
{
    return dims_.at(ndim);
}

mv::dim_type mv::Shape::dim(byte_type ndim) const
{
    return dims_.at(ndim);
}

mv::unsigned_type mv::Shape::totalSize() const
{

    unsigned_type result = dims_[0];

    for (byte_type i = 1; i < dims_.length(); ++i)
        result *= dims_[i];

    return result;
    
}

mv::dim_type& mv::Shape::operator[](int_type ndim)
{
    if (ndim < 0)
        return dim(dims_.length() + ndim);
    return dim(ndim);
}

mv::dim_type mv::Shape::operator[](int_type ndim) const
{
    if (ndim < 0)
        return dim(dims_.length() + ndim);
    return dim(ndim);
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

mv::string mv::Shape::toString() const
{

    string output("(");

    for (byte_type i = 0; i < dims_.length() - 1; ++i)
    {
        output += std::to_string(dims_[i]);
        output += ", ";
    }

    output += std::to_string(dims_[dims_.length() - 1]);
    output += ")";
    
    return output;

}

mv::json::Value mv::Shape::toJsonValue() const
{

    mv::json::Object obj;
    mv::json::Array arr;

    for (byte_type i = 0; i < dims_.length(); ++i)
    {
        arr.append(mv::Jsonable::toJsonValue(dims_[i]));
    }

    obj["rank"] = mv::Jsonable::toJsonValue(dims_.length());
    obj["dimensions"] = arr;

    return mv::json::Value(obj);

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

    for (int_type i = 1; i <= sS->ndims(); ++i)
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

mv::Shape mv::Shape::augment(const Shape& s, byte_type ndims)
{
    
    if (ndims <= s.ndims())
        return s;

    Shape sAug(ndims);
                    
    for (int i = 0; i < ndims - s.ndims(); ++i)
        sAug[i] = 1;

    for (unsigned i = 0; i < s.ndims(); ++i)
        sAug[i +  ndims - s.ndims()] = s[i];

    return sAug;

}
