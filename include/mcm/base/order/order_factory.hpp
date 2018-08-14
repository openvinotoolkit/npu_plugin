#ifndef ORDERFACTORY_HPP
#define ORDERFACTORY_HPP

#include "mcm/base/order/order.hpp"
#include "mcm/computation/model/types.hpp"


namespace mv
{
    class OrderFactory
    {
        public:
            static std::unique_ptr<OrderClass> createOrder(Order value);
    };
}

#endif
