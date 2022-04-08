//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

//
// ResolveDMAWithSwizzlingPass
//

class ResolveDMAWithSwizzlingPass final : public VPUIP::ResolveDMAWithSwizzlingBase<ResolveDMAWithSwizzlingPass> {
public:
    explicit ResolveDMAWithSwizzlingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void createNewFlatAlignedConst(Const::DeclareOp& constOp) {
    auto ctx = constOp.getContext();
    const auto contentAttr = constOp.contentAttr();
    const auto transformations = contentAttr.getTransformations();
    const auto swizzleTransformationIt =
            std::find_if(transformations.rbegin(), transformations.rend(), [](Const::TransformAttrInterface tr) {
                return tr.isa<Const::SwizzleConstantAttr>();
            });
    if (swizzleTransformationIt == transformations.rend()) {
        return;
    }
    const auto swizzleTransformation = swizzleTransformationIt->dyn_cast<Const::SwizzleConstantAttr>();

    const auto alignSize = mlir::BoolAttr::get(ctx, true);
    const auto newSwizzleTransformation = Const::SwizzleConstantAttr::get(swizzleTransformation.getSwizzleKey(),
                                                                          swizzleTransformation.getArch(), alignSize);

    SmallVector<Const::TransformAttrInterface> newTransformations;
    for (auto tr : transformations) {
        if (tr.isa<Const::SwizzleConstantAttr>()) {
            newTransformations.push_back(newSwizzleTransformation);
            continue;
        }
        newTransformations.push_back(tr);
    }

    auto newOutputType = contentAttr.getBaseContent().getType().cast<vpux::NDTypeInterface>();
    for (auto tr : newTransformations) {
        newOutputType = tr.inferOutputType(newOutputType);
    }

    const auto newTransformationsRaw = to_small_vector(
            newTransformations | transformed([](vpux::Const::TransformAttrInterface attr) -> mlir::Attribute {
                return attr;
            }));
    const auto newTransformationsAttr = mlir::ArrayAttr::get(ctx, newTransformationsRaw);

    auto newContentAttr = Const::ContentAttr::get(contentAttr.getBaseContent(), newTransformationsAttr, newOutputType);
    constOp.contentAttr(newContentAttr);

    auto constType = constOp.output().getType().cast<vpux::NDTypeInterface>();
    constType = constType.changeShapeElemType(newOutputType.getShape(), newOutputType.getElementType());

    constOp.output().setType(constType);
}

void ResolveDMAWithSwizzlingPass::safeRunOnFunc() {
    auto func = getFunction();

    func->walk([&](VPURT::TaskOp taskOp) {
        if (taskOp.getExecutorKind() != VPU::ExecutorKind::DMA_NN) {
            return;
        }

        auto dmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(taskOp.getInnerTaskOp());
        if (dmaOp == nullptr) {
            return;
        }

        auto inputOp = dmaOp.input().getDefiningOp();
        auto outputBuff = dmaOp.output_buff().getDefiningOp<VPURT::DeclareBufferOp>();

        if (inputOp == nullptr || outputBuff == nullptr) {
            return;
        }

        const auto inputBuffType = dmaOp.input().getType().cast<vpux::NDTypeInterface>();
        const auto outputBuffType = dmaOp.output_buff().getType().cast<vpux::NDTypeInterface>();

        auto inputSwizzling = getSwizzlingKey(inputBuffType);
        auto outputSwizzling = getSwizzlingKey(outputBuffType);

        VPUX_THROW_WHEN(inputSwizzling != outputSwizzling, "Incompatible swizzling setting on task - '{0}'", taskOp);

        if (inputSwizzling == 0) {
            return;
        }

        // If total allocation size is same as the one resulting from buffer shape and strides
        // then no modification is needed as DMA src/dst represent total swizzled buffer
        const auto memShape = inputBuffType.getMemShape();
        const auto memStrides = inputBuffType.getMemStrides();

        const auto outputMemShape = outputBuffType.getMemShape();
        const auto outputMemStrides = outputBuffType.getMemStrides();

        const auto buffAllocSize = inputBuffType.getTotalAllocSize().count();
        const auto outputBuffAllocSize = outputBuffType.getTotalAllocSize().count();
        const auto buffRealSize = Byte(memStrides.front() * memShape.front()).count();
        const auto outputBuffRealSize = Byte(outputMemStrides.front() * outputMemShape.front()).count();

        const auto inputAligned = (buffRealSize == buffAllocSize);
        const auto outputAligned = (outputBuffRealSize == outputBuffAllocSize);

        if (inputAligned && outputAligned) {
            _log.trace("DMA already represent total size of swizzled buffer, dmaOp - '{0}'", taskOp->getLoc());
            return;
        }

        _log.trace("Identified DMA task with swizzled buffers that require conversion to flat tensor, dmaOp - '{0}', "
                   "real size - '{1}', allocation size '{2}'",
                   taskOp->getLoc(), buffRealSize, buffAllocSize);

        auto newInputBuffType = inputBuffType;
        auto newOutputBuffType = outputBuffType;

        auto newElementType = getUInt8Type(taskOp->getContext());
        if (inputBuffType.getElemTypeSize().count() != 1) {
            newInputBuffType = newInputBuffType.changeElemType(newElementType);
            newOutputBuffType = newOutputBuffType.changeElemType(newElementType);
        }

        // Create new shape in the form of flat tensor that will represent total swizzled buffer
        // of size that is explicitly aligned to 512 as required by HW
        SmallVector<int64_t> newShape = {buffAllocSize, 1, 1, 1};
        if (inputBuffType.getElemTypeSize().count() == 1) {
            newShape = {buffAllocSize * CHAR_BIT, 1, 1, 1};
        }
        newInputBuffType = newInputBuffType.changeShape(ShapeRef(newShape));
        newOutputBuffType = newOutputBuffType.changeShape(ShapeRef(newShape));

        mlir::OpBuilder builder(taskOp.getOperation());

        mlir::Operation* newInputOp;

        builder.setInsertionPoint(inputOp);
        if (auto inputBuff = dmaOp.input().getDefiningOp<VPURT::DeclareBufferOp>()) {
            // Create new source flat buffer with aligned size
            auto newInputBuff = builder.create<VPURT::DeclareBufferOp>(
                    appendLoc(inputBuff->getLoc(), "_flat_buffer_alloc"), newInputBuffType, inputBuff.sectionAttr(),
                    inputBuff.sectionIndexAttr(), inputBuff.byteOffsetAttr(), inputBuff.swizzlingKeyAttr());
            _log.nest().trace("Create new source flat buffer allocation of shape - '{0}', op - '{1}'",
                              ShapeRef(newShape), newInputBuff->getLoc());
            newInputOp = newInputBuff.getOperation();
        } else if (auto inputCst = dmaOp.input().getDefiningOp<Const::DeclareOp>()) {
            // Create new constant with flat shape with aligned size
            createNewFlatAlignedConst(inputCst);
            _log.nest().trace("Make constant of flat aligned shape - '{0}', op - '{1}'", ShapeRef(newShape),
                              inputCst->getLoc());
            newInputOp = inputCst.getOperation();
        } else {
            VPUX_THROW("Unsupported swizzled source operand of DMA '{0}'", inputOp->getName());
        }

        // Create new destination flat buffer
        builder.setInsertionPoint(outputBuff);
        auto newOutputBuff = builder.create<VPURT::DeclareBufferOp>(
                appendLoc(outputBuff->getLoc(), "_flat_buffer_alloc"), newOutputBuffType, outputBuff.sectionAttr(),
                outputBuff.sectionIndexAttr(), outputBuff.byteOffsetAttr(), outputBuff.swizzlingKeyAttr());
        _log.nest().trace("Create new destination flat buffer allocation of shape - '{0}', op - '{1}'",
                          ShapeRef(newShape), newOutputBuff->getLoc());

        builder.setInsertionPoint(dmaOp);
        auto newDmaOp = builder.create<VPUIP::NNDMAOp>(appendLoc(dmaOp->getLoc(), "_flat_buffer_dma"),
                                                       newInputOp->getResult(0), newOutputBuff, dmaOp.portAttr(),
                                                       dmaOp.is_out_of_orderAttr(), dmaOp.is_criticalAttr());
        _log.nest().trace("Create new DMA - '{0}'", newDmaOp->getLoc());

        dmaOp->erase();
        if (inputOp->getUsers().empty()) {
            inputOp->erase();
        }
        if (outputBuff->getUsers().empty()) {
            outputBuff->erase();
        }
    });
}

}  // namespace

//
// createResolveDMAWithSwizzlingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createResolveDMAWithSwizzlingPass(Logger log) {
    return std::make_unique<ResolveDMAWithSwizzlingPass>(log);
}
