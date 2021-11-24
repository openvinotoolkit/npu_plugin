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

#include "vpux/compiler/dialect/IERT/ops.hpp"

mlir::LogicalResult vpux::IERT::TimestampOp::inferReturnTypes(mlir::MLIRContext*, llvm::Optional<mlir::Location>,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::RegionRange,
                                                              llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    IERT::TimestampOpAdaptor timestampOp(operands, attrs);
    inferredReturnTypes.push_back(timestampOp.output_buff().getType());

    return mlir::success();
}