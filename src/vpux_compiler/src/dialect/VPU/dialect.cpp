//
// Copyright Intel Corporation.
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

#include "vpux/compiler/dialect/VPU/dialect.hpp"

#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

//
// initialize
//

void vpux::VPU::VPUDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPU/generated/ops.cpp.inc>
            >();

    registerTypes();
}

//
// Generated
//

#include <vpux/compiler/dialect/VPU/generated/dialect.cpp.inc>
