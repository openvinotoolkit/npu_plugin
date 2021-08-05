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

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux {

//
// convertFunc
//

using CvtOpBuilderCb = FuncRef<mlir::Operation*(mlir::OpBuilder&, mlir::Location, mlir::Value, mlir::Type)>;

mlir::LogicalResult convertFunc(mlir::FuncOp funcOp, ArrayRef<mlir::Type> newArgTypes,
                                ArrayRef<mlir::Type> newResultTypes, CvtOpBuilderCb cvtOpBuilder,
                                Logger log = Logger::global());

//
// getDefaultGreedyRewriteConfig
//

mlir::GreedyRewriteConfig getDefaultGreedyRewriteConfig();

//
// appendLoc
//

mlir::Location appendLoc(mlir::Location baseLoc, StringRef suffix);

//
// dummyConverter
//

template <class ConcreteType>
mlir::Value dummyConverter(mlir::OpBuilder& builder, ConcreteType type, mlir::ValueRange inputs, mlir::Location loc) {
    SmallVector<mlir::Value> results;
    builder.createOrFold<mlir::UnrealizedConversionCastOp>(results, loc, type, inputs);
    return results.front();
}

//
// BufferizeTypeConverter
//

class BufferizeTypeConverter : public mlir::TypeConverter {
public:
    BufferizeTypeConverter();
};

//
// populateBufferizeMaterializationLegality
//

void populateBufferizeMaterializationLegality(mlir::ConversionTarget& target);

}  // namespace vpux
