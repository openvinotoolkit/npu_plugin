#ifndef ROWMAJOR_HPP
#define ROWMAJOR_HPP

#include "order.hpp"

namespace mv
{

    class RowMajor : public OrderClass
    {

    public:

        ~RowMajor();
        int previousContiguousDimensionIndex(const Shape& s, unsigned current_dim) const;
        int nextContiguousDimensionIndex(const Shape& s, unsigned current_dim) const;

        unsigned lastContiguousDimensionIndex(const Shape &s) const;
        unsigned firstContiguousDimensionIndex(const Shape &s) const;
    };

}

#endif
