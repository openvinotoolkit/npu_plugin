//
// Copyright 2021 Intel Corporation.
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

#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#ifdef ENABLE_PLAIDML
#include "pmlc/util/matchers.h"
#endif

#include "vpux/compiler/edsl/passes.hpp"

using namespace mlir;          // NOLINT
using namespace mlir::math;    // NOLINT
using namespace mlir::memref;  // NOLINT

namespace vpux {
namespace edsl {

#ifdef ENABLE_PLAIDML

struct RsqrtPattern final : public OpRewritePattern<DivFOp> {
    using OpRewritePattern<DivFOp>::OpRewritePattern;

    // Convert 'x / sqrt(y)' into 'x * rsqrt(y)'.
    LogicalResult matchAndRewrite(DivFOp op, PatternRewriter& rewriter) const {
        Value value;
        auto scalarPattern = m_Op<SqrtOp>(m_Capture(&value));
        if (matchPattern(op.rhs(), scalarPattern)) {
            auto rsqrt = rewriter.create<RsqrtOp>(op.getLoc(), value.getType(), value);
            rewriter.replaceOpWithNewOp<MulFOp>(op, op.getType(), op.lhs(), rsqrt);
            return success();
        }
        if (matchPattern(op.rhs(), m_Op<vector::BroadcastOp>(scalarPattern))) {
            auto rsqrt = rewriter.create<RsqrtOp>(op.getLoc(), value.getType(), value);
            auto broadcast = rewriter.create<vector::BroadcastOp>(op.getLoc(), op.getType(), rsqrt);
            rewriter.replaceOpWithNewOp<MulFOp>(op, op.getType(), op.lhs(), broadcast);
            return success();
        }
        return failure();
    }
};

struct MvuDotPattern : public OpRewritePattern<vector::ReductionOp> {
    using OpRewritePattern<vector::ReductionOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(vector::ReductionOp op, PatternRewriter& rewriter) const {
        if (op.kind() != "add") {
            return op.emitRemark("Reduction kind != 'add'");
        }

        if (!op.getType().isF32()) {
            return op.emitRemark("Reduction result type is not f32");
        }

        Value value1, value2;
        auto pattern = m_Op<MulFOp>(m_Op<FPExtOp>(m_Capture(&value1)), m_Op<FPExtOp>(m_Capture(&value2)));
        if (!matchPattern(op.vector(), pattern)) {
            return op.emitRemark("Pattern not found");
        }

        auto funcType =
                FunctionType::get(op.getContext(), ArrayRef<Type>{value1.getType(), value2.getType()}, op.getType());
        auto callee = getOrInsertFuncOp("mvuDot", funcType, op, rewriter);

        rewriter.replaceOpWithNewOp<CallOp>(op, callee, ArrayRef<Value>{value1, value2});

        return success();
    }

    FuncOp getOrInsertFuncOp(StringRef funcName, FunctionType funcType, Operation* op,
                             PatternRewriter& rewriter) const {
        Operation* funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcName);
        if (funcOp) {
            return cast<FuncOp>(*funcOp);
        }

        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(op->getParentOfType<FuncOp>());
        auto newFuncOp = rewriter.create<FuncOp>(op->getLoc(), funcName, funcType);
        newFuncOp.setPrivate();
        return newFuncOp;
    }
};

template <typename OpTy>
struct DevectorizePattern : public OpRewritePattern<OpTy> {
    using OpRewritePattern<OpTy>::OpRewritePattern;

    // convert vector op to scalar exp in for loop
    LogicalResult matchAndRewrite(OpTy op, PatternRewriter& rewriter) const {
        Value sourceVec = op.getOperand();
        if (!sourceVec.getType().isa<VectorType>()) {
            return op->emitRemark(op.getOperationName() + " has no vector type operand");
        }

        Value value;
        auto pattern = m_Op<OpTy>(m_Capture(&value));
        if (matchPattern(op.getOperation(), pattern)) {
            auto loc = op->getLoc();
            auto vectorType = value.getType().cast<VectorType>();
            auto memrefType = MemRefType::get(vectorType.getShape(), vectorType.getElementType());
            auto vecMem = rewriter.create<AllocOp>(loc, memrefType);
            SmallVector<Value, 8> vectorIndices{rewriter.create<ConstantIndexOp>(loc, 0)};
            rewriter.create<vector::TransferWriteOp>(loc, value, vecMem, vectorIndices);

            auto start = rewriter.create<ConstantIndexOp>(loc, 0);
            auto end = rewriter.create<ConstantIndexOp>(loc, vectorType.getShape()[0]);
            auto step = rewriter.create<ConstantIndexOp>(loc, 1);

            auto loopOp = rewriter.create<scf::ForOp>(loc, start, end, step);
            rewriter.setInsertionPointToStart(loopOp.getBody());
            auto index = loopOp.getInductionVar();
            auto elementIndices = SmallVector<Value, 1>{index};
            auto element = rewriter.create<LoadOp>(loc, vecMem, elementIndices);
            auto expResult = rewriter.create<OpTy>(loc, element).getResult();
            rewriter.create<StoreOp>(loc, expResult, vecMem, elementIndices);

            rewriter.setInsertionPointAfter(loopOp);
            rewriter.replaceOpWithNewOp<vector::TransferReadOp>(op, vectorType, vecMem, vectorIndices);
            return success();
        }
        return failure();
    }
};

#endif

struct ShavePatternsPass : public ShavePatternsBase<ShavePatternsPass> {
public:
    void runOnFunction() override {
#ifdef ENABLE_PLAIDML
        auto* context = &getContext();
        RewritePatternSet patterns(context);

        patterns.insert<                    //
                MvuDotPattern,              //
                RsqrtPattern,               //
                DevectorizePattern<ExpOp>,  //
                DevectorizePattern<TanhOp>>(context);
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            VPUX_THROW("ShavePatternsPass was failed.");
        }
#else
        VPUX_THROW("ShavePatternsPass is only supported when ENABLE_PLAIDML=ON");
#endif
    }
};

std::unique_ptr<mlir::Pass> createShavePatternsPass() {
    return std::make_unique<ShavePatternsPass>();
}

}  // namespace edsl
}  // namespace vpux
