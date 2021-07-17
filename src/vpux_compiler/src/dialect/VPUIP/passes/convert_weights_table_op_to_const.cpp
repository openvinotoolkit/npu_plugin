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

#include <vpux/compiler/core/aliases_info.hpp>
#include <vpux/compiler/dialect/VPUIP/blob_reader.hpp>
#include <vpux/compiler/dialect/VPUIP/nce_invariant.hpp>
#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/utils/core/enums.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

constexpr int32_t MTL_SPARSITY = 0xffffffff;

int32_t toFixedPoint(float realVal) {
    // FIXME: 2 ^ 16 might be more obvious
    return std::lround(realVal * 65536.f);
}

int32_t toHex(float realVal) {
    union f32toint32 {
        int32_t m_i32;
        float m_f32;
    };

    f32toint32 biasVal;
    biasVal.m_f32 = realVal;
    return biasVal.m_i32;
}

using BiasConverterCb = int32_t (*)(float);
const EnumMap<VPUIP::ArchKind, BiasConverterCb> biasConvertersMap = {
        {VPUIP::ArchKind::VPU3400_A0, toFixedPoint},  //
        {VPUIP::ArchKind::VPU3400, toFixedPoint},     //
        {VPUIP::ArchKind::VPU3700, toFixedPoint},     //
        {VPUIP::ArchKind::VPU3900, toFixedPoint},     //
        {VPUIP::ArchKind::VPU3720, toHex},            //
};

constexpr int32_t getKMBScale() {
    constexpr int32_t PRELU_SCALE_OFFSET = 0;
    constexpr int32_t PRELU_SCALE_VALUE = 1;

    // FIXME: PPE shift is actually 6 bit long, 2 higher bits stand for rounding mode
    constexpr int32_t PPE_SHIFT_OFFSET = 8;
    constexpr int32_t PPE_SHIFT_VALUE = 0;

    constexpr int32_t PPE_MULT_OFFSET = 16;
    // FIXME: PPE multiplier has sign, which may affect lower bits
    constexpr int32_t PPE_MULT_VALUE = 1;

    constexpr int32_t KMB_SCALE = (PRELU_SCALE_VALUE << PRELU_SCALE_OFFSET) | (PPE_SHIFT_VALUE << PPE_SHIFT_OFFSET) |
                                  (PPE_MULT_VALUE << PPE_MULT_OFFSET);

    return KMB_SCALE;
}

int32_t getMTLScale() {
    constexpr float MTL_SCALE = 1.0f;

    return toHex(MTL_SCALE);
}

using PPEConverterCb = int32_t (*)();
const EnumMap<VPUIP::ArchKind, PPEConverterCb> ppeConvertersMap = {
        {VPUIP::ArchKind::VPU3400_A0, getKMBScale},  //
        {VPUIP::ArchKind::VPU3400, getKMBScale},     //
        {VPUIP::ArchKind::VPU3700, getKMBScale},     //
        {VPUIP::ArchKind::VPU3900, getKMBScale},     //
        {VPUIP::ArchKind::VPU3720, getMTLScale},     //
};

using GetBiasCb = FuncRef<float(int64_t)>;

std::vector<int32_t> getWeightsTable(int64_t OC, GetBiasCb getBiasFP, int32_t weightPtrOffset, int32_t weightPtrStep,
                                     int32_t sparsityPtrOffset, vpux::VPUIP::ArchKind arch) {
    const auto ppeConverter = ppeConvertersMap.at(arch);
    const int32_t multShift = ppeConverter();

    const int32_t sparsityPtr = arch == VPUIP::ArchKind::VPU3720 ? MTL_SPARSITY : sparsityPtrOffset;

    const auto convertBias = [&](int64_t oc) -> int32_t {
        const auto biasVal = getBiasFP(oc);
        const auto biasConverter = biasConvertersMap.at(arch);
        return biasConverter(biasVal);
    };

    std::vector<int32_t> weightsTableVals(OC * VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC, 0);

    for (auto oc : irange(checked_cast<size_t>(OC))) {
        const auto wtInd = oc * static_cast<size_t>(VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC);

        weightsTableVals[wtInd + 0] = weightPtrOffset;
        weightsTableVals[wtInd + 1] = sparsityPtr;
        weightsTableVals[wtInd + 2] = multShift;
        weightsTableVals[wtInd + 3] = convertBias(oc);

        weightPtrOffset += weightPtrStep;
    }

    return weightsTableVals;
}

llvm::unique_function<float(int64_t)> getBiasFunc(mlir::Value bias) {
    if (bias == nullptr) {
        return [](int64_t) -> float {
            return 0.0f;
        };
    }

    auto biasConst = bias.getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_UNLESS(biasConst != nullptr, "Only constant biases are supported, got '{0}'", bias);

    auto biasContent = biasConst.content();

    return [biasContent = std::move(biasContent)](int64_t oc) -> float {
        return biasContent.getValues<float>()[oc];
    };
}

int64_t getOC(VPUIP::WeightsTableOp createWTableOp) {
    if (createWTableOp.weights() != nullptr && createWTableOp.activation_window() != nullptr) {
        // Depthwise convolution case. Weights table contains both activation window and weights.
        // FIXME the logic repeats row-major convolution
        const auto filterShape = getShape(createWTableOp.weights());
        return filterShape[IERT::ConvolutionOp::filter_out_channel_dim()];
    }

    if (createWTableOp.weights() != nullptr) {
        const auto filterShape = getShape(createWTableOp.weights());
        return filterShape[IERT::ConvolutionOp::filter_out_channel_dim()];
    }

    const auto fakeSparsity = getShape(createWTableOp.activation_window());
    return fakeSparsity[IERT::ConvolutionOp::filter_out_channel_dim()];  // actually this is input channel
}

int32_t getWeightPtrStep(VPUIP::WeightsTableOp createWTableOp) {
    if (createWTableOp.weights() != nullptr) {
        const auto filterShape = getShape(createWTableOp.weights());

        const auto IC = filterShape[IERT::ConvolutionOp::filter_in_channel_dim()];
        const auto KY = filterShape[IERT::ConvolutionOp::filter_spatial_height_dim()];
        const auto KX = filterShape[IERT::ConvolutionOp::filter_spatial_width_dim()];
        const auto eltSize = Byte(getElemTypeSize(createWTableOp.weights().getType())).count();
        if (createWTableOp.activation_window() != nullptr) {
            // Depthwise convolution case.
            // Weights table contains both activation window and weights.
            // Check that weights have expected alignment.
            // Other than that, weight step is the same for both z-major (OYXI) and depthwise convolutions.
            const auto origFilterType = createWTableOp.weights().getType().cast<mlir::ShapedType>();
            const auto depthwiseConvAlignment =
                    VPUIP::NCEInvariant::getChannelAlignment(origFilterType.getElementType());
            const int64_t weightsElementCount = IC * KY * KX;
            VPUX_THROW_UNLESS(weightsElementCount % depthwiseConvAlignment == 0,
                              "Depthwise convolution weights size must be a multiple of {0}, got {1}",
                              depthwiseConvAlignment, weightsElementCount);
        }

        return checked_cast<int32_t>(IC * KY * KX * eltSize);
    }

    return 0;
}

int32_t getTensorPtrOffset(mlir::Value input, const AliasesInfo* aliasInfo) {
    if (input == nullptr) {
        return 0;
    }

    auto output_buff = aliasInfo->getRoot(input);
    auto tensor = output_buff.getDefiningOp<IERT::StaticAllocOp>();
    VPUX_THROW_UNLESS(tensor != nullptr, "Cannot get offset");
    return checked_cast<int32_t>(tensor.offset());
}

//
// CreateWTableOpsConverter
//

class CreateWTableOpsConverter final : public mlir::OpRewritePattern<VPUIP::WeightsTableOp> {
public:
    CreateWTableOpsConverter(mlir::MLIRContext* ctx, Logger log, vpux::VPUIP::ArchKind arch,
                             const AliasesInfo* aliasInfo)
            : mlir::OpRewritePattern<VPUIP::WeightsTableOp>(ctx), _log(log), _arch(arch), _aliasInfo(aliasInfo) {
        VPUX_THROW_UNLESS(_aliasInfo != nullptr, "Got NULL pointer for AliasesInfo in ViewLikeRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::WeightsTableOp createWTableOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    vpux::VPUIP::ArchKind _arch;
    const AliasesInfo* _aliasInfo = nullptr;
};

mlir::LogicalResult CreateWTableOpsConverter::matchAndRewrite(VPUIP::WeightsTableOp createWTableOp,
                                                              mlir::PatternRewriter& rewriter) const {
    const auto OC = getOC(createWTableOp);

    int32_t weightPtrOffset = getTensorPtrOffset(createWTableOp.weights(), _aliasInfo);
    const auto weightPtrStep = getWeightPtrStep(createWTableOp);

    int32_t sparsityPtrOffset = getTensorPtrOffset(createWTableOp.activation_window(), _aliasInfo);

    auto getBiasFP = getBiasFunc(createWTableOp.bias());

    const auto weightsTable = getWeightsTable(OC, getBiasFP, weightPtrOffset, weightPtrStep, sparsityPtrOffset, _arch);

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

    const auto arch = VPUIP::getArch(module);

    VPUX_THROW_UNLESS(biasConvertersMap.find(arch) != biasConvertersMap.end(),
                      "Failed to map bias converter to target arch");
    VPUX_THROW_UNLESS(ppeConvertersMap.find(arch) != ppeConvertersMap.end(),
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
