//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU37XX/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

using namespace vpux;

//
// initialize
//

void vpux::VPU37XX::VPU37XXDialect::initialize() {
    registerTypes();
}

//
// Generated
//

#include <vpux/compiler/dialect/VPU37XX/generated/dialect.cpp.inc>
