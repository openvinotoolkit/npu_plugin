#ifndef ORDERCLASS_HPP
#define ORDERCLASS_HPP

#include <vector>
#include "include/mcm/computation/model/types.hpp"
#include "include/mcm/computation/tensor/shape.hpp"
#include "include/mcm/base/exception/shape_error.hpp"

namespace mv
{

    class OrderClass
    {

    public:

        virtual ~OrderClass()
        {

        }

        virtual int previousContiguousDimensionIndex(const Shape& s, std::size_t current_dim) const = 0;
        virtual int nextContiguousDimensionIndex(const Shape& s, std::size_t current_dim) const = 0;
        virtual std::size_t lastContiguousDimensionIndex(const Shape &s) const = 0;
        virtual std::size_t firstContiguousDimensionIndex(const Shape &s) const = 0;
        bool isLastContiguousDimensionIndex(const Shape &s, std::size_t index) const;
        bool isFirstContiguousDimensionIndex(const Shape &s, std::size_t index) const;
        unsigned subToInd(const Shape &s, const std::vector<std::size_t>& sub) const;
        std::vector<std::size_t> indToSub(const Shape &s, std::size_t idx) const;

    };

}

#endif
