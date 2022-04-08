//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIPDPU/dialect.hpp"

#include "vpux/compiler/dialect/VPUIPDPU/ops.hpp"

using namespace vpux;

//
// initialize
//

void vpux::VPUIPDPU::VPUIPDPUDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPUIPDPU/generated/ops.cpp.inc>
            >();

    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/dialect/VPUIPDPU/generated/types.cpp.inc>
            >();
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUIPDPU/generated/dialect.cpp.inc>
