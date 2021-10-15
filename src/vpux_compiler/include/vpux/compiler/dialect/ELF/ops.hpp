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

#pragma once

#include "vpux/compiler/core/ops_interfaces.hpp"
#include "vpux/compiler/dialect/ELF/attributes/enums.hpp"  // 2021_10_14
#include "vpux/compiler/dialect/ELF/types.hpp"             // 2021_10_01
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"  // Alex
#include "vpux/compiler/dialect/const/ops.hpp"

#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
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

#include <vpux/compiler/dialect/ELF/generated/dialect.hpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/ELF/generated/ops.hpp.inc>

//
// Operation verifiers
//

namespace vpux {
namespace ELF {

// mlir::LogicalResult verifyOp(DPUTaskOp op);
// mlir::LogicalResult verifyPostOp(mlir::Operation* op);

}  // namespace ELF
}  // namespace vpux

//
// Template methods
//

namespace vpux {
namespace ELF {

}  // namespace ELF
}  // namespace vpux
