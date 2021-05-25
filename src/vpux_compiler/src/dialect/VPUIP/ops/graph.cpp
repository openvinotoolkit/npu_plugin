//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

using namespace vpux;

VPUIP::GraphOp vpux::VPUIP::GraphOp::getFromModule(mlir::ModuleOp module) {
    auto graphOps = to_small_vector(module.getOps<GraphOp>());

    VPUX_THROW_UNLESS(graphOps.size() == 1, "Can't have more than one 'VPUIP::GraphOp' Operation in Module, got '{0}'",
                      graphOps.size());

    return graphOps.front();
}
