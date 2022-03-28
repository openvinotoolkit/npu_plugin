//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/type_interfaces.hpp"

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

using CvtOpBuilderCb = FuncRef<mlir::Operation*(mlir::OpBuilder&, mlir::Location, mlir::Value, vpux::NDTypeInterface)>;

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

//
// BufferizeWithDistributedTypeConverter
//

class BufferizeWithDistributedTypeConverter : public BufferizeTypeConverter {
public:
    BufferizeWithDistributedTypeConverter();
};

//
// populateBufferizeWithDistributedMaterializationLegality
//

void populateBufferizeWithDistributedMaterializationLegality(mlir::ConversionTarget& target);

//
// inferReturnTypes
//

enum class InferShapedTypeMode : uint32_t {
    SHAPE = 1 << 0,
    ELEM_TYPE = 1 << 1,
    LAYOUT = 1 << 2,
    MEM_SPACE = 1 << 3,
    SPARSITY = 1 << 4,

    ALL = std::numeric_limits<uint32_t>::max()
};

inline InferShapedTypeMode operator|(InferShapedTypeMode lhs, InferShapedTypeMode rhs) {
    return static_cast<InferShapedTypeMode>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}
inline InferShapedTypeMode operator&(InferShapedTypeMode lhs, InferShapedTypeMode rhs) {
    return static_cast<InferShapedTypeMode>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}
inline bool bitEnumContains(InferShapedTypeMode bits, InferShapedTypeMode bit) {
    return (static_cast<uint32_t>(bits) & static_cast<uint32_t>(bit)) != 0;
}

void inferReturnTypes(mlir::Operation* op, InferShapedTypeMode mode);

}  // namespace vpux
