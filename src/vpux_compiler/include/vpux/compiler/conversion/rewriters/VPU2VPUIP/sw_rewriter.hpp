//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <vpux/utils/core/logger.hpp>
#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux {

class LayerRewrite final : public mlir::ConversionPattern {
public:
    LayerRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::ConversionPattern(typeConverter, mlir::Pattern::MatchAnyOpTypeTag{}, benefitLow, ctx), _log(log) {
        setDebugName("LayerRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::Operation* origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    template <class InLayerOp>
    static mlir::Operation* dispatch(mlir::Operation* origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b);

private:
    Logger _log;
};

}  // namespace vpux
