//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/ops_interfaces.hpp"
#include "vpux/compiler/dialect/ELF/attributes.hpp"
#include "vpux/compiler/dialect/ELF/dialect.hpp"
#include "vpux/compiler/dialect/ELF/metadata.hpp"
#include "vpux/compiler/dialect/ELF/ops_interfaces.hpp"
#include "vpux/compiler/dialect/ELF/types.hpp"
#include "vpux/compiler/dialect/VPUIP/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/types.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CopyOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/ELF/ops.hpp.inc>
