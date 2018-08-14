#ifndef COLMAJOR_HPP
#define COLMAJOR_HPP

#include "order.hpp"

namespace mv
{

    class ColMajor : public OrderClass
    {

    public:

        ~ColMajor();
        //Return -1 if there is no extra dimension, dimension index otherwise
        int previousContiguousDimensionIndex(const Shape& s, unsigned current_dim) const;
        int nextContiguousDimensionIndex(const Shape& s, unsigned current_dim) const;

        unsigned lastContiguousDimensionIndex(const Shape &s) const;
        unsigned firstContiguousDimensionIndex(const Shape &s) const;

        //unsigned subToInd(const Shape &s, const static_vector<dim_type, byte_type, max_ndims>& sub) const;
        //static_vector<dim_type, byte_type, max_ndims> indToSub(const Shape &s, unsigned index) const;
    };

}

#endif
