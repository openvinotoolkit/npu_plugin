#ifndef MV_ORDER_REGISTRY_HPP_
#define MV_ORDER_REGISTRY_HPP_

#include <string>
#include "include/mcm/base/registry.hpp"
#include "include/mcm/tensor/order/order_entry.hpp"

namespace mv
{

    class OrderRegistry : public Registry<OrderRegistry, std::string, OrderEntry>
    {


    public:

        static OrderRegistry& instance();


        inline static bool checkOrder(const std::string& order_string)
        {
            return instance().find(order_string) != nullptr;
        }

        inline static const std::vector<std::size_t>& getContVector(const std::string& order_string)
        {
            return instance().find(order_string)->getContiguityVector();
        }

    };

    #define MV_REGISTER_ORDER(Name)                          \
        MV_REGISTER_ENTRY(OrderRegistry, std::string, OrderEntry, #Name)    \


}

#endif // MV_ORDER_REGISTRY_HPP_
