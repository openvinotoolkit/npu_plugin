//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
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
