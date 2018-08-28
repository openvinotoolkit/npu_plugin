#ifndef COLMAJOR_HPP
#define COLMAJOR_HPP

#include "order.hpp"

namespace mv
{

    class ColMajor : public OrderClass
    {

    public:

        ~ColMajor();
        int previousContiguousDimensionIndex(const Shape& s, std::size_t current_dim) const;
        int nextContiguousDimensionIndex(const Shape& s, std::size_t current_dim) const;

        std::size_t lastContiguousDimensionIndex(const Shape &s) const;
        std::size_t firstContiguousDimensionIndex(const Shape &s) const;
    };

}

#endif
