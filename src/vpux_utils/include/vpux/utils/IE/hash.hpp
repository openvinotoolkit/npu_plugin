//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/utils/core/hash.hpp"

#include <ngraph/node_output.hpp>

//
// Hash
//

namespace std {

template <>
struct hash<ngraph::Output<ngraph::Node>> final {
    size_t operator()(const ngraph::Output<ngraph::Node>& out) const {
        return vpux::getHash(out.get_node(), out.get_index());
    }
};

}  // namespace std
