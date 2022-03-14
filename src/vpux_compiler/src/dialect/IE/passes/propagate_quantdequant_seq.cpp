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

#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

namespace {

#ifdef RESHAPE
struct MinDimension {
    std::size_t& shapeIdx;
    ArrayRef<int64_t> shape;
    int64_t largeDimQuotient;

    MinDimension(std::size_t& shapeIdx, ArrayRef<int64_t> shape, const int64_t largeDimQuotient)
            : shapeIdx(shapeIdx), shape(shape), largeDimQuotient(largeDimQuotient){};
};

void handleConsecutiveOnes(ArrayRef<int64_t> inShape, ArrayRef<int64_t> outShape, std::size_t& startIn,
                           std::size_t& startOut, SmallVector<SmallVector<int64_t>>& reassociationVec) {
    std::size_t endIn = startIn;
    while (endIn < inShape.size() && inShape[endIn] == 1)
        endIn++;

    std::size_t endOut = startOut;
    while (endOut < outShape.size() && outShape[endOut] == 1)
        endOut++;

    for (; startIn < endIn && startOut < endOut; ++startIn, ++startOut) {
        reassociationVec[startIn].push_back(static_cast<int64_t>(startOut));
    }

    while (startIn < endIn) {
        reassociationVec[startIn].push_back(static_cast<int64_t>(startOut - 1));
        startIn++;
    }

    while (startOut < endOut) {
        reassociationVec[startIn - 1].push_back(static_cast<int64_t>(startOut));
        startOut++;
    }
}

//
// QuantDequantPropagateSeq
//
// Note: When having dims equal to 1 in one of the shapes that do not have a corresponding 1 in the other shape, there
// might be multiple dim associations possible. The current algorithm takes only one into consideration.
// E.g.: 1 x 2 x 2 x 1 x 2 x 3 -> 1 x 4 x 6 has 2 possible mappings:
//      {0} -> {0}, {1, 2, 3} -> {1}, {4, 5} -> {2} (this one is computed by the fcn below)
//      {0} -> {0}, {1, 2} -> {1}, {3, 4, 5} -> {2}
mlir::FailureOr<SmallVector<SmallVector<int64_t>>> getReassociationMap(ArrayRef<int64_t> inShape,
                                                                       ArrayRef<int64_t> outShape) {
    const auto inSize = inShape.size();
    const auto outSize = outShape.size();

    const auto nextDimIsOne = [](ArrayRef<int64_t> shape, const std::size_t index) -> bool {
        return index + 1 < shape.size() && shape[index + 1] == 1;
    };

    SmallVector<SmallVector<int64_t>> reassociationVec(inSize);
    std::size_t inIdx = 0, outIdx = 0;
    for (; inIdx < inSize && outIdx < outSize; ++inIdx, ++outIdx) {
        if (inShape[inIdx] == 1 && outShape[outIdx] == 1) {
            // Pair dims equal to 1 that have corresponding dims in the other shape
            handleConsecutiveOnes(inShape, outShape, inIdx, outIdx, reassociationVec);

            if (inIdx >= inSize || outIdx >= outSize)
                break;
        }

        // If both dims are equal, pick the one that has a dim of 1 after it. If there is no corresponding dim equal to
        // 1 in the other shape, the mapping dim_large = 1 x dim_small will be added. Without that extra condition,
        // there could be cases where that extra 1 remains floating, leading the algorithm to decide that there is no
        // valid mapping between shapes.
        const bool isInputSmallerDim = inShape[inIdx] < outShape[outIdx] ||
                                       (inShape[inIdx] == outShape[outIdx] && nextDimIsOne(inShape, inIdx));
        auto minimum = isInputSmallerDim ? MinDimension(inIdx, inShape, outShape[outIdx])
                                         : MinDimension(outIdx, outShape, inShape[inIdx]);

        do {
            if (minimum.largeDimQuotient % minimum.shape[minimum.shapeIdx] != 0)
                return mlir::failure();

            reassociationVec[inIdx].push_back(static_cast<int64_t>(outIdx));

            minimum.largeDimQuotient /= minimum.shape[minimum.shapeIdx];

            if (minimum.largeDimQuotient == 1) {
                // Exit loop if the next dim isn't 1 or if there are 1s on next dim of both shapes
                if (!nextDimIsOne(minimum.shape, minimum.shapeIdx) ||
                    (nextDimIsOne(inShape, inIdx) && nextDimIsOne(outShape, outIdx))) {
                    break;
                }
            }

            ++minimum.shapeIdx;
        } while (minimum.shapeIdx < minimum.shape.size());
    }

    // One of the shapes has trailing 1s that cannot be the result of decomposing the last dim of the other shape
    if (inIdx < inSize || outIdx < outSize)
        return mlir::failure();

    return reassociationVec;
}
#endif

class QuantDequantPropagateSeq final : public IE::QuantDequantPropagateSeqBase<QuantDequantPropagateSeq> {
public:
    explicit QuantDequantPropagateSeq(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void QuantDequantPropagateSeq::safeRunOnFunc() {
    auto func = getFunction();

    func.walk([this](vpux::IE::PermuteCastOp lastPermCast) {
        auto lastAffineReshape = lastPermCast.getOperand().getDefiningOp<IE::AffineReshapeOp>();
        if (lastAffineReshape == nullptr) {
            return;
        }

        auto lastMemPerm = lastAffineReshape.getOperand().getDefiningOp<IE::MemPermuteOp>();
        if (lastMemPerm == nullptr) {
            return;
        }

        auto sliceOp = lastMemPerm.getOperand().getDefiningOp<IE::SliceOp>();
        if (sliceOp == nullptr) {
            return;
        }

        auto secondAndOp = sliceOp.getOperand().getDefiningOp<IE::AndOp>();
        if (secondAndOp == nullptr) {
            return;
        }

        auto firstAndOp = secondAndOp.getOperand(0).getDefiningOp<IE::AndOp>();
        if (firstAndOp == nullptr) {
            return;
        }
        for (const auto& user : firstAndOp->getUsers()) {
            if (mlir::dyn_cast<vpux::IE::AndOp>(user) != secondAndOp) {
                return;
            }
        }

        auto memPerm = firstAndOp.getOperand(0).getDefiningOp<IE::MemPermuteOp>();
        if (memPerm == nullptr) {
            return;
        }
        auto expand = memPerm->getOperand(0).getDefiningOp<IE::ExpandOp>();
        if (expand == nullptr) {
            return;
        }
        auto affineReshape = expand->getOperand(0).getDefiningOp<IE::AffineReshapeOp>();
        if (affineReshape == nullptr) {
            return;
        }
        auto firstMemPerm = affineReshape->getOperand(0).getDefiningOp<IE::MemPermuteOp>();
        if (firstMemPerm == nullptr) {
            return;
        }
        auto convOp = firstMemPerm->getOperand(0).getDefiningOp<IE::ConvolutionOp>();
        if (convOp == nullptr) {
            return;
        }

        auto lastConvOp = mlir::dyn_cast<vpux::IE::ConvolutionOp>(*lastPermCast->getUsers().begin());
        if (lastConvOp == nullptr) {
            return;
        }
#define NEWNADOP
#ifdef NEWNADOP
        mlir::OpBuilder andOpBuilder(firstMemPerm);
        auto inOp = convOp;
        const auto ndType = (*firstAndOp->getResultTypes().begin()).dyn_cast<vpux::NDTypeInterface>();
        const auto tensor = firstMemPerm.getOutputs().front().getType().cast<mlir::RankedTensorType>();
        const auto newType = vpux::getTensorType(
                (*convOp->getResultTypes().begin()).dyn_cast<vpux::NDTypeInterface>().getShape(),
                ndType.getElementType(), ndType.getDimsOrder(), ndType.getMemSpace(), IE::isSparse(tensor));

        auto newQuantOp = andOpBuilder.create<IE::AndOp>(firstAndOp->getLoc(), newType, inOp.output(), inOp.output(),
                                                         firstAndOp.auto_broadcastAttr(), firstAndOp.post_opAttr());
        auto newDequantOp = andOpBuilder.create<IE::AndOp>(secondAndOp->getLoc(), inOp->getResultTypes().front(),
                                                           newQuantOp.output(), newQuantOp.output(),
                                                           firstAndOp.auto_broadcastAttr(), firstAndOp.post_opAttr());
        firstMemPerm.setOperand(newDequantOp);
#endif

#ifdef RESHAPE
        mlir::OpBuilder reshapeBuilder(lastConvOp);
        auto reshapeInOp = lastPermCast;
        const auto reshapeTensor = reshapeInOp.getOutputs().front().getType().cast<mlir::RankedTensorType>();
        const auto inShape = reshapeInOp.getOutputs().front().getType().cast<mlir::ShapedType>().getShape();
        // const auto type = tensor.cast<vpux::NDTypeInterface>();
        const auto outShapeDims = SmallVector<int64_t>{1, 64, 2, 4};
        // const auto newType = vpux::getTensorType(Shape{outShapeDims}, type.getElementType(), type.getDimsOrder(),
        //                                         type.getMemSpace(), IE::isSparse(tensor));

        auto forwardReshapeOp = reshapeBuilder.create<IE::AffineReshapeOp>(
                reshapeInOp->getLoc(), /* newType,*/ reshapeInOp.output(),
                getIntArrayOfArray(&getContext(), getReassociationMap(inShape, outShapeDims).getValue()),
                getIntArrayAttr(&getContext(), Shape{outShapeDims}));

        auto inOp2 = forwardReshapeOp;
        // const auto tensor2 = inOp2.getOutputs().front().getType().cast<mlir::RankedTensorType>();
        const auto inShape2 = inOp2.getOutputs().front().getType().cast<mlir::ShapedType>().getShape();
        // const auto type2 = tensor2.cast<vpux::NDTypeInterface>();
        const auto outShapeDims2 = SmallVector<int64_t>{1, 64, 1, 8};
        // const auto newType2 = vpux::getTensorType(Shape{outShapeDims2}, type2.getElementType(), type2.getDimsOrder(),
        //                                          type2.getMemSpace(), IE::isSparse(tensor2));

        auto reverseReshapeOp = reshapeBuilder.create<IE::AffineReshapeOp>(
                inOp2->getLoc(), /*newType2, */ inOp2.output(),
                getIntArrayOfArray(&getContext(), getReassociationMap(inShape2, outShapeDims2).getValue()),
                getIntArrayAttr(&getContext(), Shape{outShapeDims2}));
        lastConvOp.setOperand(0, reverseReshapeOp.output());
#endif
        lastAffineReshape.setOperand(affineReshape.output());
#if 0
        llvm::errs() << "------------------------------------------"
                     << "\n";
        convOp->dump();
        firstMemPerm->dump();
        llvm::errs() << firstMemPerm->getLoc() << "\n";
        affineReshape->dump();
        expand->dump();
        memPerm->dump();
        firstAndOp->dump();
        secondAndOp->dump();
        sliceOp->dump();
        lastMemPerm->dump();
        lastAffineReshape->dump();
        lastPermCast->dump();
#ifdef RESHAPE
        forwardReshapeOp->dump();
        reverseReshapeOp->dump();
#endif
        lastConvOp->dump();
#ifdef NEWNADOP
        newQuantOp->dump();
        newDequantOp->dump();
#endif
        llvm::errs() << "------------------------------------------"
                     << "\n";
#endif
    });
}

}  // namespace

//
// createQuantDequantPropagateSeq
//

std::unique_ptr<mlir::Pass> vpux::IE::createQuantDequantPropagateSeqPass(Logger log) {
    return std::make_unique<QuantDequantPropagateSeq>(log);
}
