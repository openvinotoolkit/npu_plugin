//
// Copyright (C) 2022 Intel Corporation.
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

#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"

using namespace vpux;

//
// initialize
//

void vpux::VPUIPRegMapped::VPUIPRegMappedDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPUIPRegMapped/generated/ops.cpp.inc>
            >();
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUIPRegMapped/generated/dialect.cpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPUIPRegMapped/generated/ops.cpp.inc>
