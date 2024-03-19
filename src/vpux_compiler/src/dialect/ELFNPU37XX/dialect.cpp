//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/ELFNPU37XX/dialect.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/ops.hpp"

//
// initialize
//

void vpux::ELFNPU37XX::ELFNPU37XXDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/ELFNPU37XX/ops.cpp.inc>
            >();

    registerTypes();
    registerAttributes();
}

//
// Generated
//

#include <vpux/compiler/dialect/ELFNPU37XX/dialect.cpp.inc>
