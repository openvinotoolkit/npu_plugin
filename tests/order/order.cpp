#include "gtest/gtest.h"
#include "include/mcm/order/order.hpp"

TEST(order, col_major1d)
{
    mv::Shape s({3});
    mv::Order order(mv::Order(mv::Order::getColMajorID(s.ndims())));

    try
    {
        for(unsigned i = order.firstContiguousDimensionIndex(); i != order.lastContiguousDimensionIndex(); i = order.nextContiguosDimensionIndex(i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;


        for(unsigned i = order.lastContiguousDimensionIndex(); i != order.firstContiguousDimensionIndex(); i = order.prevContiguosDimensionIndex(i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

//TODO: Order tests must be rewritten using the above template
