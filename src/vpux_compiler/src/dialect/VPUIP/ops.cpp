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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"

#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

//
// LayerWithPostOpModel
//

bool isSupportedHWPostOp(mlir::Operation* postOp) {
    if (!mlir::isa<IE::ScaleShiftOp, IE::ReLUOp, IE::ClampOp, IE::SigmoidOp, IE::TanhOp>(postOp)) {
        return false;
    }

    if (auto clampOp = mlir::dyn_cast<IE::ClampOp>(postOp)) {
        const auto minVal = clampOp.minAttr().getValueAsDouble();
        if (!isDoubleEqual(minVal, 0.0)) {
            return false;
        }

        // TODO: should be check maxVal?
    }

    return true;
}

template <class MainOpType>
class LayerWithPostOpModel final :
        public IE::LayerWithPostOpInterface::ExternalModel<LayerWithPostOpModel<MainOpType>, MainOpType> {
public:
    bool isSupportedPostOp(mlir::Operation* mainOp, mlir::Operation* postOp) const {
        if (mlir::isa<IE::ScaleShiftOp>(postOp)) {
            return true;
        }

        if (VPU::getCompilationMode(postOp) == VPU::CompilationMode::ReferenceSW) {
            return false;
        }

        if (!isSupportedHWPostOp(postOp)) {
            return false;
        }

        return VPUIP::NCEInvariant::verifyKernel(mlir::cast<MainOpType>(mainOp)).succeeded();
    }
};

//
// AlignedChannelsOpModel
//

template <class MainOpType>
class AlignedChannelsOpModel final :
        public IE::AlignedChannelsOpInterface::ExternalModel<AlignedChannelsOpModel<MainOpType>, MainOpType> {
public:
    mlir::LogicalResult verifyChannels(mlir::Operation* op) const {
        if (!canBeExecutedOnNCE(op)) {
            // SW version of the operation has no specific requirements
            return mlir::success();
        }

        return VPUIP::NCEInvariant::verifyChannels(mlir::cast<MainOpType>(op));
    }

    int64_t getChannelAlignment(mlir::Operation* op) const {
        if (!canBeExecutedOnNCE(op)) {
            // SW version of the operation has no specific requirements
            return 1;
        }

        const auto inputType = op->getOperand(0).getType().cast<mlir::ShapedType>();
        return VPUIP::NCEInvariant::getChannelAlignment(inputType.getElementType());
    }

    bool checkChannelRestrictions(mlir::Operation* op, const int64_t channels) const {
        if (!canBeExecutedOnNCE(op)) {
            // there are no such restrictions in SW mode
            return true;
        }

        const auto module = op->getParentOfType<mlir::ModuleOp>();
        const auto arch = VPU::getArch(module);

        if (arch == VPU::ArchKind::MTL && (mlir::isa<IE::MaxPoolOp>(op) || mlir::isa<IE::GroupConvolutionOp>(op))) {
            // HW restrictions for channel number
            static const SmallVector<int64_t> availiableChannels = {16, 32, 64};
            return std::find(availiableChannels.begin(), availiableChannels.end(), channels) !=
                   availiableChannels.end();
        }

        return true;
    }

private:
    static bool canBeExecutedOnNCE(mlir::Operation* op) {
        if (VPU::getCompilationMode(op) == VPU::CompilationMode::ReferenceSW) {
            // We are in reference SW compilation mode
            return false;
        }

        if (VPUIP::NCEInvariant::verifyKernel(mlir::cast<MainOpType>(op)).failed()) {
            // Basic NCE invariants check failed, the operation will fallback to SW mode
            return false;
        }

        return true;
    }
};

//
// TilingInfoOpModel
//

bool isSupportedTiling(IE::ConvolutionOp origOp, const OutputTiling& tiles, Logger log) {
    const auto inputType = origOp.input().getType().cast<mlir::ShapedType>();
    const auto filterType = origOp.filter().getType().cast<mlir::ShapedType>();
    const auto outputType = origOp.output().getType().cast<mlir::ShapedType>();

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        const auto origInputShape = getShape(origOp.input());
        const auto origFilterShape = getShape(origOp.filter());
        const auto origBiasShape = origOp.bias() != nullptr ? getShape(origOp.bias()) : ShapeRef();

        const auto inputTiling = backInferConvTile(outputTile, origInputShape, origFilterShape, origBiasShape,
                                                   origOp.strides(), origOp.pads_begin(), origOp.pads_end());

        const auto& tileConf = inputTiling.tiles;
        VPUX_THROW_UNLESS(tileConf.size() > 1, "Missed tile information. Got {0} tiles info, must be at least 2",
                          tileConf.size());
        const auto& inputTile = tileConf[0];
        const auto& filterTile = tileConf[1];

        const auto inputTileType = getDenseTileType(inputType, inputTile.offsets, inputTile.shape);
        const auto filterTileType = getDenseTileType(filterType, filterTile.offsets, filterTile.shape);
        const auto outputTileType = getDenseTileType(outputType, outputTile.offsets, outputTile.shape);

        return mlir::succeeded(VPUIP::NCEInvariant::verifyConvCMX(origOp->getLoc(),
                                                                  origOp->getParentOfType<mlir::ModuleOp>(),
                                                                  inputTileType, filterTileType, outputTileType, log));
    });
}

bool isSupportedTiling(IE::GroupConvolutionOp origOp, const OutputTiling& tiles, Logger log) {
    const auto inputType = origOp.input().getType().cast<mlir::ShapedType>();
    const auto filterType = origOp.filter().getType().cast<mlir::ShapedType>();
    const auto outputType = origOp.output().getType().cast<mlir::ShapedType>();

    auto channelsInfo = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(origOp.getOperation());

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        if (channelsInfo != nullptr && !channelsInfo.checkChannelRestrictions(outputTile.shape[Dims4D::Act::C])) {
            return false;
        }

        const auto origInputShape = getShape(origOp.input());
        const auto origFilterShape = getShape(origOp.filter());
        const auto origBiasShape = origOp.bias() != nullptr ? getShape(origOp.bias()) : ShapeRef();

        const auto inputTiling = backInferGroupConvTile(outputTile, origInputShape, origFilterShape, origBiasShape,
                                                        origOp.strides(), origOp.pads_begin(), origOp.pads_end());

        const auto& tileConf = inputTiling.tiles;
        VPUX_THROW_UNLESS(tileConf.size() > 1, "Missed tile information. Got {0} tiles info, must be at least 2",
                          tileConf.size());
        const auto& inputTile = tileConf[0];
        const auto& filterTile = tileConf[1];

        const auto inputTileType = getDenseTileType(inputType, inputTile.offsets, inputTile.shape);
        const auto filterTileType = getDenseTileType(filterType, filterTile.offsets, filterTile.shape);
        const auto outputTileType = getDenseTileType(outputType, outputTile.offsets, outputTile.shape);

        return mlir::succeeded(VPUIP::NCEInvariant::verifyGroupConvCMX(
                origOp->getLoc(), origOp->getParentOfType<mlir::ModuleOp>(), inputTileType, filterTileType,
                outputTileType, origOp.strides(), log));
    });
}

bool isSupportedTiling(IE::MaxPoolOp origOp, const OutputTiling& tiles, Logger log) {
    const auto inputType = origOp.input().getType().cast<mlir::ShapedType>();
    const auto outputType = origOp.output().getType().cast<mlir::ShapedType>();

    auto channelsInfo = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(origOp.getOperation());

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        if (channelsInfo != nullptr && !channelsInfo.checkChannelRestrictions(outputTile.shape[Dims4D::Act::C])) {
            return false;
        }

        const auto origInputShape = getShape(origOp.input());

        const auto inputTiling = backInferPoolTile(outputTile, origInputShape, origOp.kernel_size(), origOp.strides(),
                                                   origOp.pads_begin(), origOp.pads_end());

        const auto& tileConf = inputTiling.tiles;
        VPUX_THROW_UNLESS(!tileConf.empty(), "Got empty tile information");
        const auto& inputTile = tileConf[0];

        const auto inputTileType = getDenseTileType(inputType, inputTile.offsets, inputTile.shape);
        const auto outputTileType = getDenseTileType(outputType, outputTile.offsets, outputTile.shape);

        return mlir::succeeded(VPUIP::NCEInvariant::verifyPoolCMX(
                origOp->getLoc(), origOp->getParentOfType<mlir::ModuleOp>(), inputTileType, outputTileType,
                origOp.kernel_size(), origOp.strides(), log));
    });
}

template <class MainOpType>
class NCETilingInfoOpModel final :
        public IE::TilingInfoOpInterface::ExternalModel<NCETilingInfoOpModel<MainOpType>, MainOpType> {
public:
    bool isSupportedTiling(mlir::Operation* origOp, const OutputTiling& tiles, Logger log) const {
        if (!isSupportedByNCE(mlir::cast<MainOpType>(origOp), log)) {
            return true;
        }

        return ::isSupportedTiling(mlir::cast<MainOpType>(origOp), tiles, log);
    }

private:
    static bool isSupportedByNCE(MainOpType op, Logger log) {
        if (VPU::getCompilationMode(op) == VPU::CompilationMode::ReferenceSW) {
            return false;
        }

        return VPUIP::NCEInvariant::verifyKernel(op, log).succeeded() &&
               VPUIP::NCEInvariant::verifyChannels(op, log).succeeded();
    }
};

template <class MainOpType>
class NCEEltwiseTilingInfoOpModel final :
        public IE::TilingInfoOpInterface::ExternalModel<NCEEltwiseTilingInfoOpModel<MainOpType>, MainOpType> {
public:
    bool isSupportedTiling(mlir::Operation* origOp, const OutputTiling& tiles, Logger log) const {
        if (!isSupportedByNCE(mlir::cast<MainOpType>(origOp), log)) {
            return true;
        }

        const auto input1Type = origOp->getOperand(0).getType().cast<mlir::ShapedType>();
        const auto input2Type = origOp->getOperand(1).getType().cast<mlir::ShapedType>();
        const auto outputType = origOp->getResult(0).getType().cast<mlir::ShapedType>();

        return llvm::all_of(tiles, [&](const TileInfo& tile) {
            const auto input1TileType = getDenseTileType(input1Type, tile.offsets, tile.shape);
            const auto input2TileType = getDenseTileType(input2Type, tile.offsets, tile.shape);
            const auto outputTileType = getDenseTileType(outputType, tile.offsets, tile.shape);

            return mlir::succeeded(
                    VPUIP::NCEInvariant::verifyEltwiseCMX(origOp->getLoc(), origOp->getParentOfType<mlir::ModuleOp>(),
                                                          input1TileType, input2TileType, outputTileType, log));
        });
    }

private:
    static bool isSupportedByNCE(MainOpType op, Logger log) {
        if (VPU::getCompilationMode(op) == VPU::CompilationMode::ReferenceSW) {
            return false;
        }

        return VPUIP::NCEInvariant::verifyKernel(op, log).succeeded() &&
               VPUIP::NCEInvariant::verifyChannels(op, log).succeeded();
    }
};

//
// LayoutInfoOpModel
//

template <class ImplOpType>
class LayoutInfoOpModelForSW final :
        public IE::LayoutInfoOpInterface::FallbackModel<LayoutInfoOpModelForSW<ImplOpType>> {
public:
    void inferLayoutInfo(mlir::Operation* origOp, IE::LayerLayoutInfo& info) const {
        ImplOpType::inferLayoutInfo(origOp, info);
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

template <class OrigOpType, class FallbackImplOpType>
class LayoutInfoOpModelForHW final :
        public IE::LayoutInfoOpInterface::ExternalModel<LayoutInfoOpModelForHW<OrigOpType, FallbackImplOpType>,
                                                        OrigOpType> {
public:
    void inferLayoutInfo(mlir::Operation* origOp, IE::LayerLayoutInfo& info) const {
        if (!canBeExecutedOnNCE(origOp)) {
            FallbackImplOpType::inferLayoutInfo(origOp, info);
            return;
        }

        VPUIP::NCEClusterTaskOp::inferLayoutInfo(origOp, info);
    }

private:
    static bool canBeExecutedOnNCE(mlir::Operation* op) {
        if (VPU::getCompilationMode(op) == VPU::CompilationMode::ReferenceSW) {
            // We are in reference SW compilation mode
            return false;
        }

        if (VPUIP::NCEInvariant::verifyKernel(mlir::cast<OrigOpType>(op)).failed()) {
            // Basic NCE invariants check failed, the operation will fallback to SW mode
            return false;
        }
        if (VPUIP::NCEInvariant::verifyChannels(mlir::cast<OrigOpType>(op)).failed()) {
            // Basic NCE invariants check failed, the operation will fallback to SW mode
            return false;
        }

        return true;
    }
};

//
// AsyncLayerOpModel
//

mlir::Attribute getExecutorForSW(mlir::Operation* origOp, uint32_t& numUnits) {
    return VPUIP::getExecutorAttr(numUnits, origOp, VPU::ExecutorKind::SHAVE_UPA);
}

mlir::Attribute getExecutorForHW(mlir::Operation* origOp, uint32_t& numUnits) {
    if (VPU::getCompilationMode(origOp) == VPU::CompilationMode::ReferenceSW) {
        return getExecutorForSW(origOp, numUnits);
    }

    if (VPUIP::NCEInvariant::verifyOp(origOp).failed()) {
        return getExecutorForSW(origOp, numUnits);
    }

    return VPUIP::getExecutorAttr(numUnits, origOp, VPU::ExecutorKind::NCE, 1);
}

class AsyncLayerOpModelForHW final : public IERT::AsyncLayerOpInterface::FallbackModel<AsyncLayerOpModelForHW> {
public:
    mlir::Attribute getExecutor(mlir::Operation* origOp, uint32_t& numUnits) const {
        return getExecutorForHW(origOp, numUnits);
    }
};

class AsyncLayerOpModelForDMA final : public IERT::AsyncLayerOpInterface::FallbackModel<AsyncLayerOpModelForDMA> {
public:
    mlir::Attribute getExecutor(mlir::Operation* origOp, uint32_t& numUnits) const {
        return VPUIP::getExecutorAttr(numUnits, origOp, VPU::ExecutorKind::DMA_NN);
    }
};

class AsyncLayerOpModelForSW final : public IERT::AsyncLayerOpInterface::FallbackModel<AsyncLayerOpModelForSW> {
public:
    mlir::Attribute getExecutor(mlir::Operation* origOp, uint32_t& numUnits) const {
        return getExecutorForSW(origOp, numUnits);
    }
};

//
// SoftwareLayerOpModel
//

class SoftwareLayerOpModel final : public IERT::SoftwareLayerOpInterface::FallbackModel<SoftwareLayerOpModel> {
public:
    IERT::KernelInfo getKernelInfo(mlir::Operation* origOp) const {
        return VPUIP::SwKernelOp::getKernelInfo(origOp);
    }
};

//
// redirectOpInterfacesForIE
//

template <template <class, class> class OpModelForHW, template <class> class OpModelForSW>
void redirectOpInterfacesForIE(mlir::DialectRegistry& registry) {
    registry.addOpInterface<IE::ConvolutionOp, OpModelForHW<IE::ConvolutionOp, VPUIP::ConvolutionUPAOp>>();
    registry.addOpInterface<IE::GroupConvolutionOp, OpModelForHW<IE::GroupConvolutionOp, VPUIP::ConvolutionUPAOp>>();
    registry.addOpInterface<IE::MaxPoolOp, OpModelForHW<IE::MaxPoolOp, VPUIP::PoolingUPAOp>>();
    registry.addOpInterface<IE::AddOp, OpModelForHW<IE::AddOp, VPUIP::EltwiseUPAOp>>();
    registry.addOpInterface<IE::MultiplyOp, OpModelForHW<IE::MultiplyOp, VPUIP::EltwiseUPAOp>>();
    registry.addOpInterface<IE::SubtractOp, OpModelForHW<IE::SubtractOp, VPUIP::EltwiseUPAOp>>();
    registry.addOpInterface<IE::AndOp, OpModelForHW<IE::AndOp, VPUIP::EltwiseUPAOp>>();

    registry.addOpInterface<IE::ConvertOp, OpModelForSW<VPUIP::ConvertUPAOp>>();
    registry.addOpInterface<IE::SoftMaxOp, OpModelForSW<VPUIP::SoftMaxUPAOp>>();
    registry.addOpInterface<IE::AvgPoolOp, OpModelForSW<VPUIP::PoolingUPAOp>>();
    registry.addOpInterface<IE::ReLUOp, OpModelForSW<VPUIP::ReLUUPAOp>>();
    registry.addOpInterface<IE::SigmoidOp, OpModelForSW<VPUIP::SigmoidUPAOp>>();
    registry.addOpInterface<IE::ClampOp, OpModelForSW<VPUIP::ClampUPAOp>>();
    registry.addOpInterface<IE::EluOp, OpModelForSW<VPUIP::EluUPAOp>>();
    registry.addOpInterface<IE::HSwishOp, OpModelForSW<VPUIP::HSwishUPAOp>>();
    registry.addOpInterface<IE::MishOp, OpModelForSW<VPUIP::MishUPAOp>>();
    registry.addOpInterface<IE::ErfOp, OpModelForSW<VPUIP::ErfUPAOp>>();
    registry.addOpInterface<IE::BroadcastOp, OpModelForSW<VPUIP::BroadcastUPAOp>>();
    registry.addOpInterface<IE::FloorOp, OpModelForSW<VPUIP::FloorUPAOp>>();
    registry.addOpInterface<IE::RoundOp, OpModelForSW<VPUIP::RoundUPAOp>>();
    registry.addOpInterface<IE::TanhOp, OpModelForSW<VPUIP::TanhUPAOp>>();
    registry.addOpInterface<IE::SqrtOp, OpModelForSW<VPUIP::SqrtUPAOp>>();
    registry.addOpInterface<IE::LogOp, OpModelForSW<VPUIP::LogUPAOp>>();
    registry.addOpInterface<IE::FakeQuantizeOp, OpModelForSW<VPUIP::FakeQuantizeUPAOp>>();
    registry.addOpInterface<IE::GatherOp, OpModelForSW<VPUIP::GatherUPAOp>>();
    registry.addOpInterface<IE::QuantizeOp, OpModelForSW<VPUIP::QuantCastUPAOp>>();
    registry.addOpInterface<IE::DequantizeOp, OpModelForSW<VPUIP::QuantCastUPAOp>>();
    registry.addOpInterface<IE::PReluOp, OpModelForSW<VPUIP::PReluUPAOp>>();
    registry.addOpInterface<IE::LeakyReluOp, OpModelForSW<VPUIP::LeakyReluUPAOp>>();
    registry.addOpInterface<IE::DivideOp, OpModelForSW<VPUIP::EltwiseUPAOp>>();
    registry.addOpInterface<IE::SquaredDifferenceOp, OpModelForSW<VPUIP::EltwiseUPAOp>>();
    registry.addOpInterface<IE::PowerOp, OpModelForSW<VPUIP::EltwiseUPAOp>>();
    registry.addOpInterface<IE::FloorModOp, OpModelForSW<VPUIP::EltwiseUPAOp>>();
    registry.addOpInterface<IE::MinimumOp, OpModelForSW<VPUIP::EltwiseUPAOp>>();
    registry.addOpInterface<IE::MaximumOp, OpModelForSW<VPUIP::EltwiseUPAOp>>();
    registry.addOpInterface<IE::SwishOp, OpModelForSW<VPUIP::SwishUPAOp>>();
    registry.addOpInterface<IE::GRNOp, OpModelForSW<VPUIP::GRNUPAOp>>();
    registry.addOpInterface<IE::LRN_IEOp, OpModelForSW<VPUIP::NormUPAOp>>();
    registry.addOpInterface<IE::ReduceMaxOp, OpModelForSW<VPUIP::ReduceUPAOp>>();
    registry.addOpInterface<IE::ReduceSumOp, OpModelForSW<VPUIP::ReduceUPAOp>>();
    registry.addOpInterface<IE::TileOp, OpModelForSW<VPUIP::PerAxisTileUPAOp>>();
    registry.addOpInterface<IE::PerAxisTileOp, OpModelForSW<VPUIP::PerAxisTileUPAOp>>();
    registry.addOpInterface<IE::NegativeOp, OpModelForSW<VPUIP::NegativeUPAOp>>();
    registry.addOpInterface<IE::ROIPoolingOp, OpModelForSW<VPUIP::ROIPoolingUPAOp>>();
    registry.addOpInterface<IE::ProposalOp, OpModelForSW<VPUIP::ProposalUPAOp>>();
    registry.addOpInterface<IE::FullyConnectedOp, OpModelForSW<VPUIP::FullyConnectedUPAOp>>();
    registry.addOpInterface<IE::DetectionOutputOp, OpModelForSW<VPUIP::DetectionOutputUPAOp>>();
    registry.addOpInterface<IE::ScaleShiftOp, OpModelForSW<VPUIP::ScaleShiftUPAOp>>();
    registry.addOpInterface<IE::ReorderOp, OpModelForSW<VPUIP::PermuteUPAOp>>();
    registry.addOpInterface<IE::CTCGreedyDecoderOp, OpModelForSW<VPUIP::CTCGreedyDecoderUPAOp>>();
    registry.addOpInterface<IE::CTCGreedyDecoderSeqLenOp, OpModelForSW<VPUIP::CTCGreedyDecoderSeqLenUPAOp>>();
    registry.addOpInterface<IE::PadOp, OpModelForSW<VPUIP::PadUPAOp>>();
    registry.addOpInterface<IE::ExpOp, OpModelForSW<VPUIP::ExpUPAOp>>();
    registry.addOpInterface<IE::InterpolateOp, OpModelForSW<VPUIP::InterpolateUPAOp>>();
    registry.addOpInterface<IE::LSTMCellOp, OpModelForSW<VPUIP::LSTMCellUPAOp>>();
    registry.addOpInterface<IE::StridedSliceOp, OpModelForSW<VPUIP::StridedSliceUPAOp>>();
    registry.addOpInterface<IE::RegionYoloOp, OpModelForSW<VPUIP::RegionYoloUPAOp>>();
    registry.addOpInterface<IE::MVNOp, OpModelForSW<VPUIP::MVNUPAOp>>();
    registry.addOpInterface<IE::LSTMSequenceOp, OpModelForSW<VPUIP::LSTMSequenceUPAOp>>();
    registry.addOpInterface<IE::MemPermuteOp, OpModelForSW<VPUIP::PermuteUPAOp>>();
    registry.addOpInterface<IE::CeilingOp, OpModelForSW<VPUIP::CeilingUPAOp>>();
    registry.addOpInterface<IE::NormalizeIEOp, OpModelForSW<VPUIP::NormalizeIEUPAOp>>();
    registry.addOpInterface<IE::EqualOp, OpModelForSW<VPUIP::EltwiseUPAOp>>();
    registry.addOpInterface<IE::LessOp, OpModelForSW<VPUIP::EltwiseUPAOp>>();
    registry.addOpInterface<IE::LessEqualOp, OpModelForSW<VPUIP::EltwiseUPAOp>>();
    registry.addOpInterface<IE::NotEqualOp, OpModelForSW<VPUIP::EltwiseUPAOp>>();
    registry.addOpInterface<IE::GreaterOp, OpModelForSW<VPUIP::EltwiseUPAOp>>();
    registry.addOpInterface<IE::GreaterEqualOp, OpModelForSW<VPUIP::EltwiseUPAOp>>();
    registry.addOpInterface<IE::AndOp, OpModelForSW<VPUIP::EltwiseUPAOp>>();
}

//
// redirectOpInterfacesForIERT
//

template <class OpModelForHW, class OpModelForDMA, class OpModelForSW>
void redirectOpInterfacesForIERT(mlir::DialectRegistry& registry) {
    registry.addOpInterface<IERT::CopyOp, OpModelForDMA>();
    registry.addOpInterface<IERT::TimestampOp, OpModelForDMA>();

    registry.addOpInterface<IERT::ConvolutionOp, OpModelForHW>();
    registry.addOpInterface<IERT::GroupConvolutionOp, OpModelForHW>();
    registry.addOpInterface<IERT::MaxPoolOp, OpModelForHW>();
    registry.addOpInterface<IERT::AddOp, OpModelForHW>();
    registry.addOpInterface<IERT::MultiplyOp, OpModelForHW>();
    registry.addOpInterface<IERT::SubtractOp, OpModelForHW>();
    registry.addOpInterface<IERT::AndOp, OpModelForHW>();

    registry.addOpInterface<IERT::ConvertOp, OpModelForSW>();
    registry.addOpInterface<IERT::SoftMaxOp, OpModelForSW>();
    registry.addOpInterface<IERT::AvgPoolOp, OpModelForSW>();
    registry.addOpInterface<IERT::ReLUOp, OpModelForSW>();
    registry.addOpInterface<IERT::SigmoidOp, OpModelForSW>();
    registry.addOpInterface<IERT::ClampOp, OpModelForSW>();
    registry.addOpInterface<IERT::EluOp, OpModelForSW>();
    registry.addOpInterface<IERT::HSwishOp, OpModelForSW>();
    registry.addOpInterface<IERT::MishOp, OpModelForSW>();
    registry.addOpInterface<IERT::ErfOp, OpModelForSW>();
    registry.addOpInterface<IERT::BroadcastOp, OpModelForSW>();
    registry.addOpInterface<IERT::FloorOp, OpModelForSW>();
    registry.addOpInterface<IERT::RoundOp, OpModelForSW>();
    registry.addOpInterface<IERT::TanhOp, OpModelForSW>();
    registry.addOpInterface<IERT::SqrtOp, OpModelForSW>();
    registry.addOpInterface<IERT::LogOp, OpModelForSW>();
    registry.addOpInterface<IERT::QuantizeOp, OpModelForSW>();
    registry.addOpInterface<IERT::DequantizeOp, OpModelForSW>();
    registry.addOpInterface<IERT::FakeQuantizeOp, OpModelForSW>();
    registry.addOpInterface<IERT::GatherOp, OpModelForSW>();
    registry.addOpInterface<IERT::GatherElementsOp, OpModelForSW>();
    registry.addOpInterface<IERT::PReluOp, OpModelForSW>();
    registry.addOpInterface<IERT::LeakyReluOp, OpModelForSW>();
    registry.addOpInterface<IERT::DivideOp, OpModelForSW>();
    registry.addOpInterface<IERT::SquaredDifferenceOp, OpModelForSW>();
    registry.addOpInterface<IERT::PowerOp, OpModelForSW>();
    registry.addOpInterface<IERT::FloorModOp, OpModelForSW>();
    registry.addOpInterface<IERT::MinimumOp, OpModelForSW>();
    registry.addOpInterface<IERT::MaximumOp, OpModelForSW>();
    registry.addOpInterface<IERT::SwishOp, OpModelForSW>();
    registry.addOpInterface<IERT::GRNOp, OpModelForSW>();
    registry.addOpInterface<IERT::LRN_IEOp, OpModelForSW>();
    registry.addOpInterface<IERT::ReduceMaxOp, OpModelForSW>();
    registry.addOpInterface<IERT::ReduceSumOp, OpModelForSW>();
    registry.addOpInterface<IERT::TileOp, OpModelForSW>();
    registry.addOpInterface<IERT::PerAxisTileOp, OpModelForSW>();
    registry.addOpInterface<IERT::NegativeOp, OpModelForSW>();
    registry.addOpInterface<IERT::ROIPoolingOp, OpModelForSW>();
    registry.addOpInterface<IERT::ROIAlignOp, OpModelForSW>();
    registry.addOpInterface<IERT::ProposalOp, OpModelForSW>();
    registry.addOpInterface<IERT::FullyConnectedOp, OpModelForSW>();
    registry.addOpInterface<IERT::DetectionOutputOp, OpModelForSW>();
    registry.addOpInterface<IERT::ScaleShiftOp, OpModelForSW>();
    registry.addOpInterface<IERT::CTCGreedyDecoderOp, OpModelForSW>();
    registry.addOpInterface<IERT::CTCGreedyDecoderSeqLenOp, OpModelForSW>();
    registry.addOpInterface<IERT::PadOp, OpModelForSW>();
    registry.addOpInterface<IERT::ExpOp, OpModelForSW>();
    registry.addOpInterface<IERT::InterpolateOp, OpModelForSW>();
    registry.addOpInterface<IERT::LSTMCellOp, OpModelForSW>();
    registry.addOpInterface<IERT::StridedSliceOp, OpModelForSW>();
    registry.addOpInterface<IERT::RegionYoloOp, OpModelForSW>();
    registry.addOpInterface<IERT::MVNOp, OpModelForSW>();
    registry.addOpInterface<IERT::LSTMSequenceOp, OpModelForSW>();
    registry.addOpInterface<IERT::MemPermuteOp, OpModelForSW>();
    registry.addOpInterface<IERT::CeilingOp, OpModelForSW>();
    registry.addOpInterface<IERT::NormalizeIEOp, OpModelForSW>();
    registry.addOpInterface<IERT::EqualOp, OpModelForSW>();
    registry.addOpInterface<IERT::LessOp, OpModelForSW>();
    registry.addOpInterface<IERT::LessEqualOp, OpModelForSW>();
    registry.addOpInterface<IERT::NotEqualOp, OpModelForSW>();
    registry.addOpInterface<IERT::GreaterOp, OpModelForSW>();
    registry.addOpInterface<IERT::GreaterEqualOp, OpModelForSW>();
    registry.addOpInterface<IERT::TopKOp, OpModelForSW>();
    registry.addOpInterface<IERT::AndOp, OpModelForSW>();
}

}  // namespace

//
// initialize
//

void vpux::VPUIP::VPUIPDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPUIP/generated/ops.cpp.inc>
            >();
}

//
// setupExtraInterfaces
//

void vpux::VPUIP::VPUIPDialect::setupExtraInterfaces(mlir::DialectRegistry& registry) {
    registry.addOpInterface<IE::ConvolutionOp, LayerWithPostOpModel<IE::ConvolutionOp>>();
    registry.addOpInterface<IE::GroupConvolutionOp, LayerWithPostOpModel<IE::GroupConvolutionOp>>();
    registry.addOpInterface<IE::MaxPoolOp, LayerWithPostOpModel<IE::MaxPoolOp>>();
    registry.addOpInterface<IE::AddOp, LayerWithPostOpModel<IE::AddOp>>();
    registry.addOpInterface<IE::MultiplyOp, LayerWithPostOpModel<IE::MultiplyOp>>();
    registry.addOpInterface<IE::SubtractOp, LayerWithPostOpModel<IE::SubtractOp>>();
    registry.addOpInterface<IE::AndOp, LayerWithPostOpModel<IE::AndOp>>();

    registry.addOpInterface<IE::ConvolutionOp, AlignedChannelsOpModel<IE::ConvolutionOp>>();
    registry.addOpInterface<IE::GroupConvolutionOp, AlignedChannelsOpModel<IE::GroupConvolutionOp>>();
    registry.addOpInterface<IE::MaxPoolOp, AlignedChannelsOpModel<IE::MaxPoolOp>>();
    registry.addOpInterface<IE::AddOp, AlignedChannelsOpModel<IE::AddOp>>();
    registry.addOpInterface<IE::MultiplyOp, AlignedChannelsOpModel<IE::MultiplyOp>>();
    registry.addOpInterface<IE::SubtractOp, AlignedChannelsOpModel<IE::SubtractOp>>();
    registry.addOpInterface<IE::AndOp, AlignedChannelsOpModel<IE::AndOp>>();

    registry.addOpInterface<IE::ConvolutionOp, NCETilingInfoOpModel<IE::ConvolutionOp>>();
    registry.addOpInterface<IE::GroupConvolutionOp, NCETilingInfoOpModel<IE::GroupConvolutionOp>>();
    registry.addOpInterface<IE::MaxPoolOp, NCETilingInfoOpModel<IE::MaxPoolOp>>();
    registry.addOpInterface<IE::AddOp, NCEEltwiseTilingInfoOpModel<IE::AddOp>>();
    registry.addOpInterface<IE::MultiplyOp, NCEEltwiseTilingInfoOpModel<IE::MultiplyOp>>();
    registry.addOpInterface<IE::SubtractOp, NCEEltwiseTilingInfoOpModel<IE::SubtractOp>>();
    registry.addOpInterface<IE::AndOp, NCEEltwiseTilingInfoOpModel<IE::AndOp>>();

    registry.addOpInterface<IERT::SigmoidOp, SoftwareLayerOpModel>();
    registry.addOpInterface<IERT::SoftMaxOp, SoftwareLayerOpModel>();
    registry.addOpInterface<IERT::HSwishOp, SoftwareLayerOpModel>();

    redirectOpInterfacesForIE<LayoutInfoOpModelForHW, LayoutInfoOpModelForSW>(registry);
    redirectOpInterfacesForIERT<AsyncLayerOpModelForHW, AsyncLayerOpModelForDMA, AsyncLayerOpModelForSW>(registry);
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUIP/generated/dialect.cpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPUIP/generated/ops.cpp.inc>
