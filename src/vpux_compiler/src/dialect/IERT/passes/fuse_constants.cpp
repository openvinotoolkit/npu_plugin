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

#include "vpux/compiler/dialect/IERT/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/IE/float16.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"

#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops_interfaces.hpp"

#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"

#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include "vpux/utils/core/range.hpp"

using namespace vpux;

namespace {

//
// FuseConstants
//

class FuseConstants final : public mlir::OpRewritePattern<VPUIP::NCEClusterTaskOp> {
public:
    FuseConstants(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::NCEClusterTaskOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::NCEClusterTaskOp nceOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

vpux::Const::DeclareOp createFusedConstant(vpux::Const::DeclareOp weights_constant,
                                           vpux::VPUIP::WeightsTableOp weights_table_op,
                                           mlir::PatternRewriter& rewriter, Logger log) {
    auto weights_size = vpux::getTotalSize(weights_constant->getOpResult(0));
    auto weights_table_size = vpux::getTotalSize(weights_table_op->getOpResult(0));

    const auto total_size = (weights_size + weights_table_size).count();
    // shape
    SmallVector<int64_t> fusedConstShape({total_size});
    mlir::Type fusedConstElemType = getInt8Type(rewriter.getContext());
    // crete type
    const auto fusedTensorType = mlir::MemRefType::get(fusedConstShape, fusedConstElemType);

    // get constant content
    auto weightsContent = weights_constant.contentAttr().fold();

    // actually raw buffer can get by this
    const auto origWeightsRawBuf = weightsContent.getRawStorageBuf().data();

    log.warning(" Element type size {0}  Complete type {1} getStorageElemType {2} getElementType {3}",
                weightsContent.getElemTypeSize().count() / 8, weights_constant.contentAttr().getType(),
                weightsContent.getStorageElemType(), weightsContent.getElementType());

    if ((weightsContent.getElemTypeSize().count() / 8) < 2) {
        log.warning("Quantized type");
    }

    /*auto weightsValues = weightsContent.getValues<float16>();
    std::vector<float16> weightsValuesBuf(0);
    for (auto value : weightsValues) {
        weightsValuesBuf.push_back(value);
    }

    // fill weights table with zeroes
    for (int i = 0; i < weights_table_size.count() / 2; ++i) {
        weightsValuesBuf.push_back(0);
    }

    auto rawWeights = reinterpret_cast<char*>(weightsValuesBuf.data()); */
    const auto rawWeightsBuffer = makeArrayRef(origWeightsRawBuf, total_size);
    bool isSplatBuffer = false;
    VPUX_THROW_UNLESS(mlir::DenseElementsAttr::isValidRawBuffer(fusedTensorType, rawWeightsBuffer, isSplatBuffer),
                      "New constant has invalid buffer");

    mlir::ElementsAttr value;
    value = mlir::DenseElementsAttr::getFromRawBuffer(fusedTensorType, rawWeightsBuffer, isSplatBuffer);
    auto fusedConstant = rewriter.create<Const::DeclareOp>(weights_constant.getLoc(), fusedTensorType,
                                                           Const::ContentAttr::get(value));

    return fusedConstant;
}

mlir::LogicalResult FuseConstants::matchAndRewrite(VPUIP::NCEClusterTaskOp nceOp,
                                                   mlir::PatternRewriter& rewriter) const {
    // 1. Find constant inputs
    mlir::SmallVector<Const::DeclareOp> declareConstOps;
    // _log.warning("NCE Op");

    auto copyWeightsOp = nceOp.weights().getDefiningOp<IERT::CopyOp>();
    auto weightsTable = nceOp.weight_table();
    if (weightsTable == nullptr) {
        return matchFailed(rewriter, nceOp, "NCE Op does not have weights table, exit.");
    }

    auto weightsTableCopyOp = weightsTable.getDefiningOp<IERT::CopyOp>();

    if (copyWeightsOp == nullptr || weightsTableCopyOp == nullptr) {
        return matchFailed(rewriter, nceOp, "Invalid NCE Op");
    }

    auto weightsConstant = copyWeightsOp.input().getDefiningOp<Const::DeclareOp>();
    auto weightsTableOp = weightsTableCopyOp.input().getDefiningOp<VPUIP::WeightsTableOp>();

    if (weightsConstant == nullptr || weightsTableOp == nullptr) {
        return matchFailed(rewriter, nceOp, "Invalid NCE Op");
    }

    rewriter.setInsertionPoint(weightsTableOp);
    // 2. Create fused constant of u8 type with size of weights + weights table
    // Fill it with weights.
    auto fusedConstant = createFusedConstant(weightsConstant, weightsTableOp, rewriter, _log);
    if (fusedConstant == nullptr) {
        return matchFailed(rewriter, nceOp, "Unsupported NCE Op");
    }
    // _log.warning("fusedConstant: {0}", fusedConstant);

    // 3. Create new AllocOp for a new constant
    auto fusedTensorType = fusedConstant.output().getType().cast<mlir::MemRefType>();
    auto fusedTensorTypeMemSpace = changeMemSpace(
            fusedTensorType, VPUIP::PhysicalMemoryAttr::get(rewriter.getContext(), VPUIP::PhysicalMemory::CMX_NN));
    auto allocOp = rewriter.create<mlir::memref::AllocOp>(weightsConstant.getLoc(), fusedTensorTypeMemSpace);
    // _log.warning("allocOp: {0}", allocOp);

    // 4. create CopyOp, copy constant to allocated buffer
    auto copyOp = rewriter.create<IERT::CopyOp>(weightsConstant.getLoc(), fusedConstant.output(), allocOp.memref());
    // _log.warning("copyOp: {0}", copyOp);

    // 5.  Replace weights constant with sequence fused_constant -> subview -> view
    // 5.1 Get U8 subtensor
    auto weights_size = vpux::getTotalSize(weightsConstant->getOpResult(0));
    SmallVector<int64_t> weightsSubtensor({weights_size.count()});
    vpux::ShapeRef weightsOffsets({0});

    auto subViewOp = rewriter.create<IERT::SubViewOp>(
            weightsConstant.getLoc(), copyOp.output(), weightsOffsets,
            vpux::ShapeRef(parseIntArrayAttr<int64_t>(getIntArrayAttr(rewriter.getContext(), weightsSubtensor))));
    // _log.warning("subviewOp: {0}", subViewOp);

    // 5.2 Reinterpret U8 memref as FP16 memref auto castViewOp =
    rewriter.replaceOpWithNewOp<IERT::ViewOp>(copyWeightsOp, subViewOp.result(), copyWeightsOp.output_buff().getType());

    /* Use mlir::memref::ViewOp instead of new view op

    auto offsetConst = rewriter.create<mlir::arith::ConstantIndexOp>(weightsConstant.getLoc(), 0);
    auto offsetConstValue = mlir::Value(offsetConst);
    _log.warning("offsetConstValue: {0}", offsetConstValue);
    // new shapes

    const auto shape = vpux::getShape(copyWeightsOp.output_buff());  // .getType().cast<mlir::ShapedType>()
    mlir::SmallVector<mlir::Value> dims;
    for (auto dim : shape) {
        auto dimValue = mlir::Value(rewriter.create<mlir::arith::ConstantIndexOp>(weightsConstant.getLoc(), dim));
        dims.push_back(dimValue);
    }
    auto resultShape = mlir::ValueRange(dims);

    auto mlirViewOp =
            rewriter.create<mlir::memref::ViewOp>(weightsConstant.getLoc(), weightsConstant->getOpResult(0).getType(),
                                                  subViewOp.result(), offsetConst, resultShape);

    _log.warning("mlirViewOp: {0}", mlirViewOp);

        getMemRefType(shape);
        auto dstOrder = DimsOrder::fromType(copyWeightsOp.output_buff().getType().cast<mlir::ShapedType>());
        auto newOrder = dstOrder.toAffineMap(getContext());
        auto permuteCastOp = rewriter.create<IERT::PermuteCastOp>(
                weightsConstant.getLoc(), copyWeightsOp.output_buff().getType(), viewOp.getResult(), newOrder,
       newOrder); _log.warning("permuteCastOp: {0}", permuteCastOp);
        */

    // _log.warning("viewOp: {0}", castViewOp);

    // 6 replace weights table input from old memref to new memref via same sequence memref -> subview -> view
    // 6.1 create subview from memref
    /*auto subViewOpWT = rewriter.create<IERT::SubViewOp>(
            weightsConstant.getLoc(), copyOp.output_buff(), weightsOffsets,
            vpux::ShapeRef(parseIntArrayAttr<int64_t>(getIntArrayAttr(rewriter.getContext(), weightsSubtensor))));*/

    // 6.2 replace old alloc with view cast op
    // auto origAllocOp = copyWeightsOp.output_buff().getDefiningOp<mlir::memref::AllocOp>();

    /* rewriter.replaceOpWithNewOp<mlir::memref::ViewOp>(origAllocOp, weightsConstant->getOpResult(0).getType(),
                                                      subViewOpWT.result(), offsetConst, resultShape); */
    // rewriter.replaceOpWithNewOp<IERT::ViewOp>(origAllocOp, subViewOpWT.result(),
    // copyWeightsOp.output_buff().getType());

    {
        // const auto copyOp = mlir::dyn_cast_or_null<IERT::CopyOp>(*(createWTableOp.output().getUsers().begin()));
        auto weightsTableCopyOp = weightsTable.getDefiningOp<IERT::CopyOp>();
        if (weightsTableCopyOp == nullptr) {
            _log.warning("copyOp == nullptr");
            return mlir::failure();
        }

        int64_t size = static_cast<int64_t>(vpux::getTotalSize(weightsTable).count());
        SmallVector<int64_t> weightsTableSize({size});
        auto weightsSize = vpux::getTotalSize(weightsTableOp.weights()).count();
        SmallVector<int64_t> weightsSizes({weightsSize});
        vpux::ShapeRef weightsTableOffsets(weightsSizes);

        auto subViewOp =
                rewriter.create<IERT::SubViewOp>(weightsTableCopyOp->getLoc(), copyOp.output(), weightsTableOffsets,
                                                 vpux::ShapeRef(parseIntArrayAttr<int64_t>(getIntArrayAttr(
                                                         rewriter.getContext(), weightsTableSize))));  // .getResult()

        const auto outType = weightsTableOp.output().getType();

        auto viewOpWT = rewriter.create<IERT::ViewOp>(copyOp->getLoc(), subViewOp.result(), outType);

        rewriter.replaceOp(weightsTableCopyOp, mlir::ValueRange({viewOpWT.getResult()}));
    }

    return mlir::success();
}

//
// FuseConstantsPass
//

class FuseConstantsPass final : public IERT::FuseConstantsBase<FuseConstantsPass> {
public:
    explicit FuseConstantsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void FuseConstantsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<FuseConstants>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeCopiesPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createFuseConstantsPass(Logger log) {
    return std::make_unique<FuseConstantsPass>(log);
}
