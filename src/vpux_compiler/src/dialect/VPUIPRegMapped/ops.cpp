//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/attributes.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/types.hpp"

using namespace vpux;

//
// initialize
//

void vpux::VPUIPRegMapped::VPUIPRegMappedDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPUIPRegMapped/generated/ops.cpp.inc>
            >();

    registerTypes();
    registerAttributes();
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUIPRegMapped/generated/dialect.cpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPUIPRegMapped/generated/ops.cpp.inc>
