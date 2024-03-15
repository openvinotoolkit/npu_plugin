//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

//
// initialize
//

void vpux::VPU::VPUDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPU/ops.cpp.inc>
            >();

    registerAttributes();
    registerTypes();
}

//
// Generated
//

#include <vpux/compiler/dialect/VPU/dialect.cpp.inc>
