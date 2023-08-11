//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>

#include "vpux/compiler/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include "vpux/compiler/dialect/VPURegMapped/attributes.hpp"
#include "vpux/compiler/dialect/VPURegMapped/enums.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

//
// Generated
//

#include <vpux/compiler/dialect/VPURegMapped/generated/dialect.hpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPURegMapped/generated/ops.hpp.inc>
