#include "include/mcm/tensor/shape.hpp"

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

    return const_cast<std::size_t&>(static_cast<const Shape*>(this)->operator[](ndim));

}

const std::size_t& mv::Shape::operator[](int ndim) const
{

    if (ndim >= static_cast<int>(dims_.size()) || dims_.size() + ndim < 0)
        throw ArgumentError(*this, "index subscript", std::to_string(ndim),
            "Exceeds the dimensionality " + std::to_string(ndims()));

    if (ndim < 0)
        return dims_[dims_.size() + ndim];

    return dims_[ndim];

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

std::string mv::Shape::getLogID() const
{
    return "Shape " + toString();
}
