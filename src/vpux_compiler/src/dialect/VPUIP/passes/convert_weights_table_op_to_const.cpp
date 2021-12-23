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

#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"

#include "vpux/utils/core/enums.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

int64_t getOC(VPUIP::WeightsTableOp createWTableOp) {
    if (createWTableOp.weights() != nullptr && createWTableOp.activation_window() != nullptr) {
        // Depthwise convolution case. Weights table contains both activation window and weights.
        // FIXME the logic repeats row-major convolution
        const auto filterShape = getShape(createWTableOp.weights());
        return filterShape[Dims4D::Filter::OC];
    }

    if (createWTableOp.weights() != nullptr) {
        const auto filterShape = getShape(createWTableOp.weights());
        return filterShape[Dims4D::Filter::OC];
    }

    const auto fakeSparsity = getShape(createWTableOp.activation_window());
    return fakeSparsity[Dims4D::Filter::OC];  // actually this is input channel
}

int32_t getWeightPtrStep(VPUIP::WeightsTableOp createWTableOp) {
    if (createWTableOp.weights() == nullptr) {
        return 0;
    }

    const auto filterShape = getShape(createWTableOp.weights());

    const auto IC = filterShape[Dims4D::Filter::IC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    if (createWTableOp.activation_window() != nullptr) {
        // Depthwise convolution case.
        // Weights table contains both activation window and weights.
        // Check that weights have expected alignment.
        // Other than that, weight step is the same for both z-major (OYXI) and depthwise convolutions.
        const auto origFilterType = createWTableOp.weights().getType().cast<mlir::ShapedType>();
        const auto depthwiseConvAlignment = VPU::NCEInvariant::getAlignment(origFilterType.getElementType());
        const auto weightsElementCount = IC * KY * KX;
        VPUX_THROW_UNLESS(weightsElementCount % depthwiseConvAlignment == 0,
                          "Depthwise convolution weights size must be a multiple of {0}, got {1}",
                          depthwiseConvAlignment, weightsElementCount);
    }

    const Byte eltSize = getElemTypeSize(createWTableOp.weights().getType());
    return checked_cast<int32_t>(IC * KY * KX * eltSize.count());
}

//
// CreateWTableOpsConverter
//

class CreateWTableOpsConverter final : public mlir::OpRewritePattern<VPUIP::WeightsTableOp> {
public:
    CreateWTableOpsConverter(mlir::MLIRContext* ctx, Logger log, VPU::ArchKind arch, const AliasesInfo* aliasInfo)
            : mlir::OpRewritePattern<VPUIP::WeightsTableOp>(ctx), _log(log), _arch(arch), _aliasInfo(aliasInfo) {
        VPUX_THROW_UNLESS(_aliasInfo != nullptr, "Got NULL pointer for AliasesInfo in ViewLikeRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::WeightsTableOp createWTableOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    VPU::ArchKind _arch;
    const AliasesInfo* _aliasInfo = nullptr;
};

mlir::LogicalResult CreateWTableOpsConverter::matchAndRewrite(VPUIP::WeightsTableOp createWTableOp,
                                                              mlir::PatternRewriter& rewriter) const {
    const auto OC = getOC(createWTableOp);
    // Actual weight and sparsity pointers are not known at this stage, so the constant
    // is filled only with offsets from the base pointers. Once the memory scheduler
    // allocates the memory and the pointers are known the transformation is added to the
    // constant. Finally the transformation shall add the base pointers to the offsets.
    const auto weightPtrOffset = 0;
    const auto sparsityPtrOffset = 0;
    const auto weightPtrStep = getWeightPtrStep(createWTableOp);

    const auto op_inElemType = createWTableOp.op_input().getType().cast<mlir::ShapedType>().getElementType();
    const auto op_outElemType = createWTableOp.op_output().getType().cast<mlir::ShapedType>().getElementType();
    const auto op_weightsElemType =
            createWTableOp.weights() ? createWTableOp.weights().getType().cast<mlir::ShapedType>().getElementType()
                                     : nullptr;
    const auto weightsTable = VPU::NCESparsity::getWeightsTable(
            op_inElemType, op_outElemType, weightPtrOffset, weightPtrStep, sparsityPtrOffset, _arch, OC,
            op_weightsElemType, createWTableOp.biasAttr(), createWTableOp.ppeAttr());

    const auto outType = createWTableOp.output().getType();
    const auto shapedType = outType.dyn_cast_or_null<mlir::ShapedType>();

    const auto dataStorageType =
            mlir::RankedTensorType::get(shapedType.getShape(), getSInt32Type(rewriter.getContext()));
    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, makeArrayRef(weightsTable));

    rewriter.replaceOpWithNewOp<Const::DeclareOp>(createWTableOp, outType, Const::ContentAttr::get(dataAttr));

    return mlir::success();
}

//
// ConvertCreateWTableOps2VPUIPPass
//

class ConvertWeightsTableOp2Const final : public VPUIP::ConvertWeightsTableOp2ConstBase<ConvertWeightsTableOp2Const> {
public:
    explicit ConvertWeightsTableOp2Const(Logger log);

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

ConvertWeightsTableOp2Const::ConvertWeightsTableOp2Const(Logger log): _log(log) {
    _log.setName(Base::getArgumentName());
}

void ConvertWeightsTableOp2Const::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();
    auto& aliasInfo = getAnalysis<AliasesInfo>();

    auto module = func->getParentOfType<mlir::ModuleOp>();

    const auto arch = VPU::getArch(module);

    VPUX_THROW_UNLESS(VPU::NCESparsity::biasConvertersMap.find(arch) != VPU::NCESparsity::biasConvertersMap.end(),
                      "Failed to map bias converter to target arch");
    VPUX_THROW_UNLESS(VPU::NCESparsity::ppeConvertersMap.find(arch) != VPU::NCESparsity::ppeConvertersMap.end(),
                      "Failed to map PPE converter to target arch");

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<VPUIP::WeightsTableOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<CreateWTableOpsConverter>(&ctx, _log, arch, &aliasInfo);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertWeightsTableOp2ConstPass(Logger log) {
    return std::make_unique<ConvertWeightsTableOp2Const>(log);
}
