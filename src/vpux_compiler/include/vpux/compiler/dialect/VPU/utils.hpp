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

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/preprocessing.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>

namespace vpux {
namespace VPU {

unsigned align(unsigned x, unsigned mult);

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
