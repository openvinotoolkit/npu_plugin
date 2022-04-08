//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIPDPU/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/types.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/types.hpp"

#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/format.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <initializer_list>

using namespace vpux;

static inline mlir::LogicalResult verifyArchKind(mlir::Operation* op,
                                                 std::initializer_list<VPU::ArchKind> supportedArchs) {
    auto arch = VPU::getArch(op);

    if (arch != VPU::ArchKind::UNKNOWN) {
        if (std::find(cbegin(supportedArchs), cend(supportedArchs), arch) == cend(supportedArchs)) {
            return errorAt(op, "op not supported in {0}", arch);
        }
    }

    return mlir::success();
}

//
// ArchKindVPUX37XX
//

mlir::LogicalResult vpux::VPUIPDPU::verifyArchKindVPUX37XX(mlir::Operation* op) {
    return verifyArchKind(op, {VPU::ArchKind::VPUX37XX});
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUIPDPU/generated/ops_interfaces.cpp.inc>
