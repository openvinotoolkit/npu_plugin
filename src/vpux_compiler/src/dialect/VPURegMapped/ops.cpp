//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

using namespace vpux;

//
// initialize
//

void vpux::VPURegMapped::VPURegMappedDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPURegMapped/generated/ops.cpp.inc>
            >();

    registerAttributes();
    registerTypes();
}

//
// Generated
//

#include <vpux/compiler/dialect/VPURegMapped/generated/dialect.cpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPURegMapped/generated/ops.cpp.inc>
