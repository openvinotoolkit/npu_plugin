//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/attributes.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/attributes/enums.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/attributes/structs.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/types.hpp"
#include "vpux/compiler/dialect/VPURT/types.hpp"
#include "vpux/compiler/dialect/VPURegMapped/enums.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CopyOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

//
// Generated
//

#include <vpux/compiler/dialect/VPUMI37XX/generated/dialect.hpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPUMI37XX/generated/ops.hpp.inc>
