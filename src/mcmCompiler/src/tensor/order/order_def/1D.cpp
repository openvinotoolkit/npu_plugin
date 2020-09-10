#include "include/mcm/tensor/order/order_registry.hpp"
#include "include/mcm/base/exception/order_error.hpp"
#include "include/mcm/tensor/order/order.hpp"

namespace mv
{
    MV_REGISTER_ORDER(W).setContiguityVector({0});
    MV_REGISTER_ORDER(N).setContiguityVector({0});
    MV_REGISTER_ORDER(C).setContiguityVector({0});
    MV_REGISTER_ORDER(H).setContiguityVector({0});
    MV_REGISTER_ORDER(T).setContiguityVector({0});

}
