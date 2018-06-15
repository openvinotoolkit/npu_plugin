#ifndef SHAPE_HPP_
#define SHAPE_HPP_

#include "include/mcm/computation/model/types.hpp"
#include "include/mcm/logger/printable.hpp"
#include "include/mcm/graph/static_vector.hpp"

namespace mv
{

    class Shape : public Printable
    {

        static_vector<dim_type, byte_type, max_ndims> dims_;

        void addDim(dim_type newDim)
        {
            dims_.push_back(newDim);
        }

        template<typename... Dims>
        void addDim(dim_type newDim, Dims... newDims)
        {
            dims_.push_back(newDim);
            addDim(newDims...);
        }

    public:

        template<typename... Dims>
        Shape(Dims... dims)
        {
            addDim(dims...);
        }

        Shape(const Shape& other) :
        dims_(other.dims_)
        {

        }

        Shape()
        {
            
        }

        byte_type ndims() const
        {
            return dims_.length();
        }

        dim_type dim(byte_type ndim) const
        {
            return dims_[ndim];
        }

        unsigned_type totalSize() const
        {

            unsigned_type result = dims_[0];

            for (byte_type i = 1; i < dims_.length(); ++i)
                result *= dims_[i];

            return result;
            
        }

        dim_type operator[](int_type ndim) const
        {
            if (ndim < 0)
                return dim(dims_.length() - ndim);
            return dim(ndim);
        }

        Shape& operator=(const Shape& other)
        {
            dims_ = other.dims_;
            return *this;
        }

        bool operator==(const Shape& other) const
        {
            return dims_ == other.dims_;
        }

        bool operator!=(const Shape& other) const
        {
            return !operator==(other);
        }

        string toString() const
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

    };

}

#endif // SHAPE_HPP_
