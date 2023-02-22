//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIPDPU/types.hpp"

#include "vpux/compiler/dialect/VPUIPDPU/ops.hpp"

#include <llvm/ADT/TypeSwitch.h>

#include <llvm/Support/Debug.h>

using namespace vpux;

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/VPUIPDPU/generated/types.cpp.inc>
#undef GET_TYPEDEF_CLASSES
