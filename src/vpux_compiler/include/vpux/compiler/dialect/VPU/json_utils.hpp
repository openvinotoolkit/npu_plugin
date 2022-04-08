//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/json.hpp"
#include "vpux/compiler/dialect/VPU/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"

namespace vpux {
namespace VPU {

Json readManualStrategyJSON(StringRef fileName);
void writeManualStrategyJSON(StringRef fileName, const Json& json);

Json convertAttrToJSON(mlir::Attribute attr);
void createStrategyJSONFromOperations(Json& json, llvm::MapVector<mlir::Location, mlir::Operation*>& operations,
                                      ArrayRef<StringRef> strategyAttributes);
mlir::Attribute convertJSONToAttr(mlir::Attribute oldAttr, const Json& newAttrVal);
void overwriteManualStrategy(Json& manualStrategy, llvm::MapVector<mlir::Location, mlir::Operation*>& operations);

}  // namespace VPU
}  // namespace vpux
