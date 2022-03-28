//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/attributes/stride_reqs.hpp"

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/preprocessing.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>

namespace vpux {
namespace VPU {

//
// DW Convolution utility
//

mlir::Value alignDepthWiseWeightsTensor(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origFilter);

//
// CM Convolution utility
//

mlir::Value alignChannelMajorWeightsTensor(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origFilter);

}  // namespace VPU
}  // namespace vpux
