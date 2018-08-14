#ifndef ROW_MAJOR_PLANAR_HPP
#define ROW_MAJOR_PLANAR_HPP

#include "order.hpp"

namespace mv
{

    class RowMajorPlanar : public OrderClass
    {

    public:

        ~RowMajorPlanar();
        int previousContiguousDimensionIndex(const Shape& s, unsigned current_dim) const;
        int nextContiguousDimensionIndex(const Shape& s, unsigned current_dim) const;

        unsigned lastContiguousDimensionIndex(const Shape &s) const;
        unsigned firstContiguousDimensionIndex(const Shape &s) const;
    };

}

#endif
