//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

// Helper class to wrap the arguments for ExpandConverter::applyPadding
class PaddingContext {
public:
    PaddingContext(const mlir::Location loc, const ShapeRef inShape, const mlir::Value expandedBuffer,
                   const mlir::Value constantBuffer)
            : _loc(loc), _inShape(inShape), _expandedBuffer(expandedBuffer), _constantBuffer(constantBuffer){};
    PaddingContext(const PaddingContext&) = delete;
    PaddingContext(const PaddingContext&&) = delete;
    PaddingContext& operator=(const PaddingContext&) = delete;
    PaddingContext& operator=(const PaddingContext&&) = delete;

    const mlir::Location _loc;
    ShapeRef _inShape;
    const mlir::Value _expandedBuffer;
    const mlir::Value _constantBuffer;
};

//
// ConvertExpandPass
//

class ConvertExpandPass final : public VPUIP::ConvertExpandBase<ConvertExpandPass> {
public:
    explicit ConvertExpandPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    mlir::Value applyPadding(const int64_t padAxis, const int64_t padValue, ArrayRef<int64_t> inSubViewOffsets,
                             const PaddingContext& padCtx, mlir::OpBuilder& builder) const;
    size_t getMaxExpandConstShape(mlir::FuncOp func, Logger log);
    Dim getPadDim(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType);
};

mlir::Value ConvertExpandPass::applyPadding(const int64_t padAxis, const int64_t padValue,
                                            ArrayRef<int64_t> inSubViewOffsets, const PaddingContext& padCtx,
                                            mlir::OpBuilder& builder) const {
    const auto& location = padCtx._loc;
    const auto& inShape = padCtx._inShape;
    const auto& expandedBuffer = padCtx._expandedBuffer;
    const auto& constantBuffer = padCtx._constantBuffer;
    SmallVector<int64_t> subViewOffsets;
    std::copy(inSubViewOffsets.begin(), inSubViewOffsets.end(), std::back_inserter(subViewOffsets));

    auto constantOp = constantBuffer.getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_UNLESS(constantOp != nullptr, "Can not get constant Op");

    const auto constShapeType = constantOp.output().getType().cast<NDTypeInterface>();
    const auto constOuputShape = constShapeType.getShape();
    Shape subViewShape;
    std::copy(inShape.begin(), inShape.end(), std::back_inserter(subViewShape));
    subViewShape[Dim(padAxis)] = padValue;
    VPUX_THROW_UNLESS(subViewShape.totalSize() <= constOuputShape.totalSize(),
                      "Constant subview shape size '{0}' large than full size '{1}'", subViewShape.totalSize(),
                      constOuputShape.totalSize());

    // Step 1: Create SubView Op to get the right constant size
    VPUX_THROW_UNLESS(constOuputShape.size() == 1, "Constant Op unexpect shape size '{0}'", constOuputShape);
    const auto constSubviewOffset = SmallVector<int64_t>(1, 0);
    const auto constSubviewShape = SmallVector<int64_t>(1, subViewShape.totalSize());
    auto constSubView =
            builder.create<VPUIP::SubViewOp>(appendLoc(location, "_constant_subview_{0}_{1}", padAxis, padValue),
                                             constantOp, constSubviewOffset, constSubviewShape);

    // Step 2: Create Reshape Op to match concat shape
    const auto shapeType = constSubView.getType().cast<vpux::NDTypeInterface>();
    const auto newShapeType = shapeType.changeShape(subViewShape);
    auto reshapeOp = builder.create<VPUIP::GenericReshapeOp>(
            appendLoc(location, "_constant_reshape_{0}_{1}", padAxis, padValue), newShapeType, constSubView.result());

    // Step 3: Creat PermuteCast Op to match concat layout
    const auto expandOutBufferType = expandedBuffer.getType().cast<NDTypeInterface>();
    const auto newLayoutType = newShapeType.changeDimsOrder(expandOutBufferType.getDimsOrder());
    const auto dstOrderAttr =
            mlir::AffineMapAttr::get(expandOutBufferType.getDimsOrder().toAffineMap(reshapeOp.getContext()));
    const auto memPermAttr = mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(reshapeOp.getContext()));
    auto permuteCastOp =
            builder.create<VPUIP::PermuteCastOp>(appendLoc(location, "_constant_permute_{0}_{1}", padAxis, padValue),
                                                 newLayoutType, reshapeOp.output(), dstOrderAttr, memPermAttr);

    // Step 4: Create Copy Op for concat concatant input
    auto subView = builder.create<VPUIP::SubViewOp>(appendLoc(location, "_expand_subview_{0}_{1}", padAxis, padValue),
                                                    expandedBuffer, Shape(subViewOffsets), subViewShape);
    auto subViewCopy = builder.create<VPUIP::CopyOp>(appendLoc(location, "_expand_copy_{0}_{1}", padAxis, padValue),
                                                     permuteCastOp.result(), subView);

    return subViewCopy.output();
}

size_t ConvertExpandPass::getMaxExpandConstShape(mlir::FuncOp func, Logger log) {
    size_t maxShapeSize = 0;
    func->walk([&](VPUIP::ExpandOp origOp) {
        auto inType = origOp.input().getType().dyn_cast<vpux::NDTypeInterface>();
        auto inShape = getShape(origOp.input());
        auto outShape = getShape(origOp.output());

        if (!inType.getElementType().isa<mlir::Float16Type>()) {
            _log.nest().trace("ExpandOp type '{0}' should with FP16 precision", inType.getElementType());
            return;
        }

        VPUX_THROW_UNLESS(outShape.totalSize() > inShape.totalSize(),
                          "Unexpect Expand input shape '{0}' output shape '{1}'", inShape, outShape);
        auto diffShapeSize = outShape.totalSize() - inShape.totalSize();
        maxShapeSize = std::max(checked_cast<size_t>(diffShapeSize), maxShapeSize);

        log.trace("Found Expand Operation '{0}' inshape '{1}' outshape '{2}', constant maxShapeSize is '{3}'",
                  origOp->getLoc(), inShape, outShape, maxShapeSize);
    });

    return maxShapeSize;
}

Dim ConvertExpandPass::getPadDim(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType) {
    const auto inShape = inType.getShape();
    const auto outShape = outType.getShape();
    const auto ioShapes = zip(inShape, outShape);
    const auto dimDiffPredicate = [](const std::tuple<int64_t, int64_t>& ioDims) -> bool {
        const auto& inDim = std::get<0>(ioDims);
        const auto& outDim = std::get<1>(ioDims);
        return inDim != outDim;
    };

    const auto diffAxisIter = std::find_if(ioShapes.begin(), ioShapes.end(), dimDiffPredicate);
    VPUX_THROW_UNLESS(diffAxisIter != ioShapes.end(), "Expand inShape '{0}' same with the outShape '{1}'", inShape,
                      outShape);

    const auto padAxis = std::distance(ioShapes.begin(), diffAxisIter);
    return Dim(padAxis);
}

//
// safeRunOnFunc
//

void ConvertExpandPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    mlir::OpBuilder builder(&func.getBody().front().front());

    // For Expand(FP16), replace the op with concat a const op
    //     input                input      const
    //       |                    \          /
    //   Expand(fp16)     =>         Concat
    //       |                         |
    // Note that only create one largest Constant Op and reuse for all Expand layers in the model
    // Always Set this Constant with 1D shape size, it is convenient to reshape for specific Expand
    // For Expand(U8), the op will be replaced by single DMA directly in later pass(ConvertToDMA) for better efficiency
    const auto constantShapeSize = checked_cast<int64_t>(getMaxExpandConstShape(func, _log));
    const auto dataStorageType = mlir::RankedTensorType::get({constantShapeSize}, mlir::Float16Type::get(&ctx));
    const auto denseElementVal = mlir::DenseElementsAttr::get(dataStorageType, checked_cast<float16>(0.f));

    auto constantOp =
            builder.create<Const::DeclareOp>(mlir::UnknownLoc::get(&ctx), vpux::convertToMemRef(dataStorageType),
                                             Const::ContentAttr::get(denseElementVal));

    func->walk([&](VPUIP::ExpandOp origOp) {
        _log.trace("Found Expand Operation '{0}'", origOp.getLoc());

        const auto inputType = origOp.input().getType().cast<vpux::NDTypeInterface>();
        const auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();
        _log.nest().trace("inType: '{0}', outType: '{1}', padBegin: '{2}', padEnd: '{3}'", inputType, outputType,
                          origOp.pads_begin(), origOp.pads_end());

        if (!inputType.getElementType().isa<mlir::Float16Type>()) {
            _log.nest().trace("ExpandOp type should with FP16 precision, but got '{0}'", inputType.getElementType());
            return;
        }

        mlir::OpBuilder builder(origOp.getOperation());
        auto newMemRefOutputType = outputType;
        auto expandedBuffer =
                builder.create<mlir::memref::AllocOp>(origOp->getLoc(), newMemRefOutputType.cast<mlir::MemRefType>());

        const auto nonZeroAxisPredicate = [](const int64_t dim) -> bool {
            return dim > 0;
        };

        SmallVector<mlir::Value> concatInputs;
        const auto inShape = inputType.getShape();
        auto subViewOffsets = SmallVector<int64_t>(inShape.size(), 0);
        PaddingContext padCtx(origOp->getLoc(), ShapeRef(inShape), expandedBuffer, constantOp.output());

        // Apply pads_begin
        _log.nest().trace("Process Expand Operation '{0}' for pads begin", origOp->getLoc());
        const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
        const auto padBeginAxisIter = std::find_if(padsBegin.begin(), padsBegin.end(), nonZeroAxisPredicate);
        if (padBeginAxisIter != padsBegin.end()) {
            const auto padBeginAxis = std::distance(padsBegin.begin(), padBeginAxisIter);
            const auto padValue = padsBegin[padBeginAxis];
            const auto padOut = applyPadding(padBeginAxis, padValue, subViewOffsets, padCtx, builder);
            concatInputs.push_back(padOut);
            subViewOffsets[padBeginAxis] += padValue;
        }

        // Copy the input with offset according to the padding in the beginning
        _log.nest().trace("Process Expand Operation '{0}' for real input data", origOp->getLoc());
        builder.setInsertionPoint(origOp);
        const auto tensorShape = to_small_vector(inShape);
        auto tensorSubView =
                builder.create<VPUIP::SubViewOp>(origOp.getLoc(), expandedBuffer, subViewOffsets, tensorShape);
        auto tensorSubViewCopy = builder.create<VPUIP::CopyOp>(origOp->getLoc(), origOp.input(), tensorSubView);

        concatInputs.push_back(tensorSubViewCopy.output());

        // Increment offsets
        const auto padAxis = getPadDim(inputType, outputType);
        subViewOffsets[padAxis.ind()] += tensorShape[padAxis.ind()];

        // Apply pads_end
        _log.nest().trace("Process Expand Operation '{0}' for pads end", origOp->getLoc());
        const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());
        const auto padEndAxisIter = std::find_if(padsEnd.begin(), padsEnd.end(), nonZeroAxisPredicate);
        if (padEndAxisIter != padsEnd.end()) {
            const auto padEndAxis = std::distance(padsEnd.begin(), padEndAxisIter);
            const auto padValue = padsEnd[padEndAxis];
            const auto padOut = applyPadding(padEndAxis, padValue, subViewOffsets, padCtx, builder);
            concatInputs.push_back(padOut);
        }

        auto concatViewOp = builder.create<VPUIP::ConcatViewOp>(origOp->getLoc(), concatInputs, expandedBuffer);
        _log.nest().trace("Create ConcatViewOp '{0}'", concatViewOp->getLoc());

        origOp->replaceAllUsesWith(concatViewOp);
        origOp->erase();
    });
}

}  // namespace

//
// createConvertExpandPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertExpandPass(Logger log) {
    return std::make_unique<ConvertExpandPass>(log);
}
