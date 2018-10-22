#ifndef MV_ORDER_REGISTRY_HPP_
#define MV_ORDER_REGISTRY_HPP_

#include <string>
#include "include/mcm/base/registry.hpp"
#include "include/mcm/order/order_entry.hpp"

namespace mv
{
    class RutimeError : public std::runtime_error
    {

    public:

        explicit RutimeError(const std::string& whatArg);

    };


    class OrderRegistry : public Registry<std::string, OrderEntry>
    {


    public:

        static OrderRegistry& instance();


        inline static bool checkOrder(const std::string& order_string)
        {
            return instance().find(order_string) != nullptr;
        }

        inline static std::vector<std::size_t>& getContVector(const std::string& order_string)
        {
            return instance().find(order_string)->getContiguityVector();
        }

    };

    #define MV_REGISTER_ORDER(Name)                          \
        MV_REGISTER_ENTRY(std::string, OrderEntry, #Name)    \


}

#endif // MV_ORDER_REGISTRY_HPP_
