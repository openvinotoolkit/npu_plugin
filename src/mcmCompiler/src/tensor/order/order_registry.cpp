//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "include/mcm/tensor/order/order_registry.hpp"

namespace mv
{
    MV_DEFINE_REGISTRY(OrderRegistry, std::string, mv::OrderEntry)
}

mv::OrderRegistry& mv::OrderRegistry::instance()
{
    return Registry<OrderRegistry, std::string, mv::OrderEntry>::instance();
}
