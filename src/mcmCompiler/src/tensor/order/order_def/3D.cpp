#include "include/mcm/tensor/order/order_registry.hpp"
#include "include/mcm/base/exception/order_error.hpp"
#include "include/mcm/tensor/order/order.hpp"

namespace mv
{
    MV_REGISTER_ORDER(WHC).setContiguityVector({2, 1, 0});
    MV_REGISTER_ORDER(HWC).setContiguityVector({2, 0, 1});
    MV_REGISTER_ORDER(WCH).setContiguityVector({1, 2, 0});
    MV_REGISTER_ORDER(HCW).setContiguityVector({0, 2, 1});
    MV_REGISTER_ORDER(CWH).setContiguityVector({1, 0, 2});
    MV_REGISTER_ORDER(CHW).setContiguityVector({0, 1, 2});
}
