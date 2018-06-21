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

        Shape(byte_type n)
        {
            for (unsigned i = 0; i < n; ++i)
                addDim(0);
        }

        Shape()
        {
            
        }

        byte_type ndims() const
        {
            return dims_.length();
        }

        dim_type& dim(byte_type ndim)
        {
            return dims_.at(ndim);
        }

        dim_type dim(byte_type ndim) const
        {
            return dims_.at(ndim);
        }

        unsigned_type totalSize() const
        {

            unsigned_type result = dims_[0];

            for (byte_type i = 1; i < dims_.length(); ++i)
                result *= dims_[i];

            return result;
            
        }

        dim_type& operator[](int_type ndim)
        {
            if (ndim < 0)
                return dim(dims_.length() + ndim);
            return dim(ndim);
        }

        dim_type operator[](int_type ndim) const
        {
            if (ndim < 0)
                return dim(dims_.length() + ndim);
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

        static Shape broadcast(const Shape& s1, const Shape& s2)
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

        static Shape augment(const Shape& s, byte_type ndims)
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

        


    };

}

#endif // SHAPE_HPP_
