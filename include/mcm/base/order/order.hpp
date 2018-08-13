#ifndef ORDERCLASS_HPP
#define ORDERCLASS_HPP

#include "mcm/computation/model/types.hpp"
#include "mcm/computation/tensor/shape.hpp"
#include "mcm/base/exception/shape_error.hpp"

namespace mv
{

    class OrderClass
    {

    public:

        virtual ~OrderClass()
        {

        }

        virtual int previousContiguousDimensionIndex(const Shape& s, unsigned current_dim) const = 0;
        virtual int nextContiguousDimensionIndex(const Shape& s, unsigned current_dim) const = 0;
        virtual bool isLastContiguousDimensionIndex(const Shape &s, unsigned index) const = 0;
        virtual bool isFirstContiguousDimensionIndex(const Shape &s, unsigned index) const = 0;
        virtual unsigned lastContiguousDimensionIndex(const Shape &s) const = 0;
        virtual unsigned firstContiguousDimensionIndex(const Shape &s) const = 0;
        virtual unsigned subToInd(const Shape &s, const static_vector<dim_type, byte_type, max_ndims>& sub) const = 0;
        virtual static_vector<dim_type, byte_type, max_ndims> indToSub(const Shape &s, unsigned index) const = 0;

    };

}

#endif
