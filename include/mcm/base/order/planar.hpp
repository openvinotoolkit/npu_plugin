#ifndef PLANAR_HPP
#define PLANAR_HPP

#include "order.hpp"

namespace mv
{

    class Planar : public OrderClass
    {

    public:

        ~Planar();
        //Return -1 if there is no extra dimension, dimension index otherwise
        int previousContiguousDimensionIndex(const Shape& s, unsigned current_dim) const;
        int nextContiguousDimensionIndex(const Shape& s, unsigned current_dim) const;

        unsigned lastContiguousDimensionIndex(const Shape &s) const;
        unsigned firstContiguousDimensionIndex(const Shape &s) const;

        //unsigned subToInd(const Shape &s, const static_vector<mv::dim_type, mv::byte_type, mv::max_ndims>& sub) const;
        //static_vector<mv::dim_type, mv::byte_type, mv::max_ndims> indToSub(const Shape &s, unsigned index) const;
    };

}

#endif
