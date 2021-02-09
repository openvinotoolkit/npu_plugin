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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

using namespace vpux;

VPUIP::GraphOp vpux::VPUIP::GraphOp::getFromModule(mlir::ModuleOp module) {
    auto graphOps = to_small_vector(module.getOps<GraphOp>());

    VPUX_THROW_UNLESS(graphOps.size() == 1, "Can't have more than one 'VPUIP::GraphOp' Operation in Module, got '{0}'",
                      graphOps.size());

    return graphOps.front();
}
