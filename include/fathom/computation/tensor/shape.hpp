#ifndef SHAPE_HPP_
#define SHAPE_HPP_

#include "include/fathom/computation/model/types.hpp"
#include "include/fathom/computation/logger/printable.hpp"

namespace mv
{

    class Shape : public Printable
    {

        byte_type ndims_;
        dim_type dims_[max_ndims];

        void addDim(byte_type dim)
        {
            dims_[ndims_] = dim;
            ++ndims_;
        }

        template<typename... Dims>
        void addDim(byte_type dim, Dims... dims)
        {
            addDim(dim);
            addDim(dims...);
        }

    public:

        template<typename... Dims>
        Shape(Dims... dims) : ndims_(0)
        { 
            addDim(dims...);
        }

        Shape(const Shape &other) :
        ndims_(other.ndims_)
        {
            for (byte_type i = 0; i < ndims_; ++i)
                dims_[i] = other.dims_[i];
        }

        byte_type ndims() const
        {
            return ndims_;
        }

        dim_type dim(byte_type ndim) const
        {
            assert(ndim < ndims_ && "Index of dimensinos exceeds number of dimensions");
            return dims_[ndim];
        }

        unsigned_type totalSize() const
        {

            unsigned_type result = dims_[0];

            for (byte_type i = 1; i < ndims_; ++i)
                result *= dims_[i];

            return result;
            
        }

        dim_type operator[](byte_type ndim) const
        {
            return dim(ndim);
        }

        Shape& operator=(const Shape &other)
        {
            ndims_ = other.ndims_;
            for (byte_type i = 0; i < ndims_; ++i)
                dims_[i] = other.dims_[i];

            return *this;
        }

        bool operator==(const Shape &other) const
        {
            if (ndims_ != other.ndims_)
                return false;

            for (byte_type i = 0; i < ndims_; ++i)
                if (dims_[i] != other.dims_[i])
                    return false;

            return true;

        }

        string toString() const
        {

            string output("(");

            for (byte_type i = 0; i < ndims_ - 1; ++i)
            {
                output += std::to_string(dims_[i]);
                output += ", ";
            }

            output += std::to_string(dims_[ndims_ - 1]);
            output += ")";
            
            return output;

        }

    };

}

#endif // SHAPE_HPP_