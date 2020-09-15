#include "include/mcm/tensor/order/order_registry.hpp"
#include "include/mcm/base/exception/order_error.hpp"
#include "include/mcm/tensor/order/order.hpp"

namespace mv
{
    MV_REGISTER_ORDER(WHCN).setContiguityVector({3, 2, 1, 0});
    MV_REGISTER_ORDER(HWCN).setContiguityVector({3, 2, 0, 1});
    MV_REGISTER_ORDER(WCHN).setContiguityVector({3, 1, 2, 0});
    MV_REGISTER_ORDER(HCWN).setContiguityVector({3, 0, 2, 1});
    MV_REGISTER_ORDER(CWHN).setContiguityVector({3, 1, 0, 2});
    MV_REGISTER_ORDER(CHWN).setContiguityVector({3, 0, 1, 2});
    MV_REGISTER_ORDER(WHNC).setContiguityVector({2, 3, 1, 0});
    MV_REGISTER_ORDER(HWNC).setContiguityVector({2, 3, 0, 1});
    MV_REGISTER_ORDER(WCNH).setContiguityVector({1, 3, 2, 0});
    MV_REGISTER_ORDER(HCNW).setContiguityVector({0, 3, 2, 1});
    MV_REGISTER_ORDER(CWNH).setContiguityVector({1, 3, 0, 2});
    MV_REGISTER_ORDER(CHNW).setContiguityVector({0, 3, 1, 2});
    MV_REGISTER_ORDER(WNHC).setContiguityVector({2, 1, 3, 0});
    MV_REGISTER_ORDER(HNWC).setContiguityVector({2, 0, 3, 1});
    MV_REGISTER_ORDER(WNCH).setContiguityVector({1, 2, 3, 0});
    MV_REGISTER_ORDER(HNCW).setContiguityVector({0, 2, 3, 1});
    MV_REGISTER_ORDER(CNWH).setContiguityVector({1, 0, 3, 2});
    MV_REGISTER_ORDER(CNHW).setContiguityVector({0, 1, 3, 2});
    MV_REGISTER_ORDER(NWHC).setContiguityVector({2, 1, 0, 3});
    MV_REGISTER_ORDER(NHWC).setContiguityVector({2, 0, 1, 3});
    MV_REGISTER_ORDER(NWCH).setContiguityVector({1, 2, 0, 3});
    MV_REGISTER_ORDER(NHCW).setContiguityVector({0, 2, 1, 3});
    MV_REGISTER_ORDER(NCWH).setContiguityVector({1, 0, 2, 3});
    MV_REGISTER_ORDER(NCHW).setContiguityVector({0, 1, 2, 3});
}
