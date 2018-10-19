#include "include/mcm/order/order_registry.hpp"

namespace mv
{
    MV_DEFINE_REGISTRY(std::string, mv::OrderEntry)
}

mv::OrderRegistry& mv::OrderRegistry::instance()
{
    return static_cast<OrderRegistry&>(Registry<std::string, OrderEntry>::instance());
}
