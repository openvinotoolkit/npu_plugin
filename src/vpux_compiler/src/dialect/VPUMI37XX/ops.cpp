//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/types.hpp"

using namespace vpux;

//
// initialize
//

void vpux::VPUMI37XX::VPUMI37XXDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPUMI37XX/generated/ops.cpp.inc>
            >();

    registerTypes();
    registerAttributes();
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUMI37XX/generated/dialect.cpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPUMI37XX/generated/ops.cpp.inc>
