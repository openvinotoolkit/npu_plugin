#ifndef COLMAJOR_HPP
#define COLMAJOR_HPP

#include "order.hpp"

namespace mv
{

    class ColMajor : public OrderClass
    {

    public:

        ~ColMajor();
        int previousContiguousDimensionIndex(const Shape& s, unsigned current_dim) const;
        int nextContiguousDimensionIndex(const Shape& s, unsigned current_dim) const;

        unsigned lastContiguousDimensionIndex(const Shape &s) const;
        unsigned firstContiguousDimensionIndex(const Shape &s) const;
    };

}

#endif
