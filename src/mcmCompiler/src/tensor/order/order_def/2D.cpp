#include "include/mcm/tensor/order/order_registry.hpp"
#include "include/mcm/base/exception/order_error.hpp"
#include "include/mcm/tensor/order/order.hpp"

namespace mv
{
    MV_REGISTER_ORDER(WH).setContiguityVector({1, 0});
    MV_REGISTER_ORDER(HW).setContiguityVector({0, 1});
    MV_REGISTER_ORDER(NC).setContiguityVector({0, 1});
    MV_REGISTER_ORDER(WC).setContiguityVector({1, 0});
}
