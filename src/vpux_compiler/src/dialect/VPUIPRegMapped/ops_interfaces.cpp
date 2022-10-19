//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIPRegMapped/ops_interfaces.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/format.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

//
// TaskOpInterface
//

void vpux::VPUIPRegMapped::getTaskEffects(mlir::Operation* op, SmallVectorImpl<MemoryEffect>& effects) {
    VPUX_UNUSED(op);
    VPUX_UNUSED(effects);

    // Note: Do VPUIPRegMapped ops have modelable effects? The question is if we leave the TaskEffects interface for
    //        RegMapped ops or not.
    //   Currently we do not use the interface, but will we need it in the future? If yes, should we leave it here, just
    //   as a scheleton,
    //        and extend it later or delete it alltogheter and re-add it once we will deffinitely use it?

    return;
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUIPRegMapped/generated/ops_interfaces.cpp.inc>
