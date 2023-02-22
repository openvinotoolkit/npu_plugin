//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/dialect/VPUIPRegMapped/types.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

mlir::Attribute getVPUIPRegMapped_RegisterFieldAttr(::mlir::MLIRContext* context,
                                                    vpux::VPUIPRegMapped::RegFieldType value);
mlir::ArrayAttr getVPUIPRegMapped_RegisterFieldArrayAttr(mlir::OpBuilder builder,
                                                         mlir::ArrayRef<vpux::VPUIPRegMapped::RegFieldType> values);

mlir::Attribute getVPUIPRegMapped_RegisterAttr(::mlir::MLIRContext* context, vpux::VPUIPRegMapped::RegisterType value);
mlir::ArrayAttr getVPUIPRegMapped_RegisterArrayAttr(mlir::OpBuilder builder,
                                                    mlir::ArrayRef<vpux::VPUIPRegMapped::RegisterType> values);

//
// Generated
//

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/VPUIPRegMapped/generated/attributes.hpp.inc>
#undef GET_ATTRDEF_CLASSES
