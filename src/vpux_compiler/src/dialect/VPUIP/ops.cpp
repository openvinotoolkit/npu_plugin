//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/core/attributes/indexed_symbol_attr.hpp"
#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/types.hpp"
#include "vpux/compiler/dialect/VPURT/types.hpp"

#include "vpux/compiler/utils/asm.hpp"
#include "vpux/compiler/utils/custom_pwl_table.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>

#include <llvm/ADT/TypeSwitch.h>
#include <vpux/compiler/dialect/IE/utils/resources.hpp>

using namespace vpux;

namespace {

//
// LayerWithPostOpModel
//

bool isSupportedHWPostOp(mlir::Operation* mainOp, mlir::Operation* postOp) {
    if (!mlir::isa<IE::ScaleShiftOp, IE::ReLUOp, IE::ClampOp, IE::SigmoidOp, IE::TanhOp, IE::LeakyReluOp, IE::PReluOp>(
                postOp)) {
        return false;
    }

    const auto module = postOp->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    const auto isQuantized = [](mlir::Operation* op, mlir::Operation* postOp) -> bool {
        auto isFakeQuantizeOpInput = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(op->getOperand(0).getDefiningOp());
        auto isFakeQuantizeOpOutput = true;
        for (auto user : postOp->getUsers()) {
            if (!mlir::isa<IE::FakeQuantizeOp>(user)) {
                isFakeQuantizeOpOutput = false;
                break;
            }
        }

        // since FusePostOps is called also after LowPrecisionPipeline
        const auto operandType = postOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
        const auto isQuantizedElemType = operandType.getElementType().isa<mlir::quant::QuantizedType>();

        return (isFakeQuantizeOpOutput && isFakeQuantizeOpInput) || isQuantizedElemType;
    };

    if (mlir::isa<IE::SigmoidOp, IE::TanhOp>(postOp)) {
        // These ops do not get fused for float cases to avoid dropping accuracy. Because PWL is not accurate for FP16
        if (arch == VPU::ArchKind::VPUX30XX) {
            if (!isQuantized(mainOp, postOp)) {
                return false;
            }
        } else if (arch == VPU::ArchKind::VPUX37XX) {
            // VPUX37XX don't support sigmoid and tanh
            return false;
        }
    }

    auto clampOp = mlir::dyn_cast<IE::ClampOp>(postOp);
    if (clampOp != nullptr) {
        const auto minVal = clampOp.minAttr().getValueAsDouble();
        if (!isDoubleEqual(minVal, 0.0) && !isQuantized(mainOp, postOp)) {
            return false;
        }

        // TODO: should be check maxVal?
    }

    // MaxPool on VPUX37XX supports only ReLU and ReLU-X.
    const bool isSpecialMaxPoolCase = arch == VPU::ArchKind::VPUX37XX;
    const bool isSupportedByMaxPool = mlir::isa<IE::ReLUOp>(postOp) || mlir::isa<IE::ClampOp>(postOp);
    if (isSpecialMaxPoolCase && mlir::isa<IE::MaxPoolOp>(mainOp) && !isSupportedByMaxPool) {
        return false;
    }

    if (auto leakyRelu = mlir::dyn_cast<IE::LeakyReluOp>(postOp)) {
        const auto getUniformQuantizedType =
                [](IE::FakeQuantizeOp fakeQuantizeOp) -> mlir::quant::UniformQuantizedType {
            if (fakeQuantizeOp == nullptr) {
                return nullptr;
            }

            auto outLoConst = fakeQuantizeOp.output_low().getDefiningOp<Const::DeclareOp>();
            auto outHiConst = fakeQuantizeOp.output_high().getDefiningOp<Const::DeclareOp>();
            const auto realType = fakeQuantizeOp.input().getType().cast<vpux::NDTypeInterface>();
            const auto realElemType = realType.getElementType().cast<mlir::FloatType>();

            const auto outElemType =
                    getQuantizedType(outLoConst.contentAttr(), outHiConst.contentAttr(), fakeQuantizeOp.levels(),
                                     realElemType, false, fakeQuantizeOp.getLoc(), fakeQuantizeOp.auto_broadcast());
            return outElemType.dyn_cast<mlir::quant::UniformQuantizedType>();
        };

        if (mlir::isa<IE::AddOp, IE::AndOp, IE::MultiplyOp>(mainOp)) {
            return false;
        }

        const auto reluSlope = leakyRelu.negative_slopeAttr().getValueAsDouble();
        if (reluSlope < 0 && arch == VPU::ArchKind::VPUX30XX) {
            return false;
        }

        if (!isQuantized(mainOp, postOp)) {
            return true;
        }

        // All conditions from below must be after (!isQuantized(mainOp, postOp)) check from above.
        // Check below is for dKMB case
        if (arch == VPU::ArchKind::VPUX37XX) {
            return true;
        }

        if (!leakyRelu.output().hasOneUse()) {
            return false;
        }

        const auto fqOp = mlir::dyn_cast<IE::FakeQuantizeOp>(*(leakyRelu.output().getUsers().begin()));
        const auto uniformElemType = getUniformQuantizedType(fqOp);
        if (uniformElemType == nullptr) {
            return false;
        }

        const auto zeroPoint = uniformElemType.getZeroPoint();
        if (isSupportedPReLU(static_cast<float>(reluSlope), zeroPoint)) {
            return true;
        }

        return false;
    }

    return true;
}

template <class MainOpType>
class LayerWithPostOpModel final :
        public IE::LayerWithPostOpInterface::ExternalModel<LayerWithPostOpModel<MainOpType>, MainOpType> {
public:
    bool isSupportedPostOp(mlir::Operation* mainOp, mlir::Operation* postOp) const {
        if (VPU::getCompilationMode(postOp) == VPU::CompilationMode::ReferenceSW) {
            return false;
        }

        if (!isSupportedHWPostOp(mainOp, postOp)) {
            return false;
        }

        return VPUIP::NCEInvariant::verifyKernel(mlir::cast<MainOpType>(mainOp)).succeeded();
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

        const auto module = op->getParentOfType<mlir::ModuleOp>();
        const auto arch = VPU::getArch(module);
        if (arch == VPU::ArchKind::VPUX37XX && mlir::isa<IE::MultiplyOp, IE::SubtractOp, IE::AndOp>(op)) {
            return false;
        }

        return true;
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

    int64_t getInputChannelAlignment(mlir::Operation* op) const {
        if (!canBeExecutedOnNCE(op)) {
            // SW version of the operation has no specific requirements
            return 1;
        }

        auto module = op->getParentOfType<mlir::ModuleOp>();

        const auto arch = VPU::getArch(module);
        const auto inputType = op->getOperand(0).getType().cast<vpux::NDTypeInterface>();

        if (mlir::isa<IE::ConvolutionOp>(op)) {
            const auto inOrder = inputType.getDimsOrder();
            if (inOrder == DimsOrder::NCHW) {
                // C-major convolution has no specific requirements
                return 1;
            }

            const auto weightsType = op->getOperand(1).getType().cast<vpux::NDTypeInterface>();
            const bool isFP16 = inputType.getElementType().isF16() || weightsType.getElementType().isF16();

            // TODO: Compress Convolution for the moment does not work with multiclustering, nor with FP16 inputs.

            // We cannot have FP16 inputs for now because at the moment we do not have a clean way of stripping
            // the weights of the padding up to 4. Without that, we cannot set the compressed conv sparsity
            // pattern for the unpadded num of channels. Therefore, we set the cm_sp_pattern to 4 and pad the
            // weights with 0s. The activation input, however, is strided, not padded with 0, so for fp16
            // we run the risk of multiplying the 0 weight with a NaN. That is not an issue for integer types.
            if ((arch == VPU::ArchKind::VPUX37XX) && !isFP16) {
                if (inputType.getShape()[Dims4D::Act::C] <= VPU::NCEInvariant::VPU_COMPRESSED_INPUT_CHANNEL_NUM) {
                    return VPU::NCEInvariant::VPU_COMPRESSED_INPUT_CHANNEL_NUM;
                }
            }
        }

        return VPU::NCEInvariant::getAlignment(inputType.getElementType());
    }
    int64_t getOutputChannelAlignment(mlir::Operation* op) const {
        if (!canBeExecutedOnNCE(op)) {
            // SW version of the operation has no specific requirements
            return 1;
        }
        const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
        return VPU::NCEInvariant::getAlignment(outputType.getElementType());
    }

private:
    static bool canBeViewLikeOp(mlir::Operation* op) {
        if (!op) {
            return false;
        }
        if (!mlir::isa<IE::SliceOp, IE::ConcatOp, IE::SplitOp, IE::ReshapeOp, IE::SqueezeOp, IE::UnsqueezeOp,
                       IE::QuantizeCastOp, IE::PermuteCastOp>(op)) {
            return false;
        }
        return true;
    }
    static bool parentCanBeExecutedOnNCE(mlir::Operation* op) {
        if (!op) {
            return false;
        }
        if (VPU::getCompilationMode(op) == VPU::CompilationMode::ReferenceSW) {
            // We are in reference SW compilation mode
            return false;
        }
        if (VPUIP::NCEInvariant::verifyKernel(op).failed()) {
            // Basic NCE invariants check failed, the operation will fallback to SW mode
            return false;
        }
        return true;
    }
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

bool isSupportedIsolatedTiling(VPU::ConvolutionOp origOp, const OutputTiling& tiles, Logger log) {
    const auto inputType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto filterType = origOp.filter().getType().cast<vpux::NDTypeInterface>();
    const auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        const auto origInputShape = getShape(origOp.input());
        const auto origFilterShape = getShape(origOp.filter());
        const auto origBiasShape = origOp.bias() != nullptr ? getShape(origOp.bias()) : ShapeRef();
        const auto origPadding = PadInfo(origOp.pads_begin(), origOp.pads_end());

        const auto inputTiling = backInferConvTile(outputTile, origInputShape, origFilterShape, origBiasShape,
                                                   origOp.strides(), origPadding);

        const auto& tileConf = inputTiling.tiles;
        VPUX_THROW_UNLESS(tileConf.size() > 1, "Missed tile information. Got {0} tiles info, must be at least 2",
                          tileConf.size());
        const auto& inputTile = tileConf[0];
        const auto& filterTile = tileConf[1];

        const auto inputTileType = inputType.extractDenseTile(inputTile.offsets, inputTile.shape);
        const auto filterTileType = filterType.extractDenseTile(filterTile.offsets, filterTile.shape);
        const auto outputTileType = outputType.extractDenseTile(outputTile.offsets, outputTile.shape);

        return mlir::succeeded(VPUIP::NCEInvariant::verifyConvCMX(
                origOp->getLoc(), origOp->getParentOfType<mlir::ModuleOp>(), inputTileType, filterTileType,
                outputTileType, origOp.strides(), log));
    });
}

bool isSupportedIsolatedTiling(VPU::NCEConvolutionOp origOp, const OutputTiling& tiles, Logger log) {
    const auto inputType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto filterType = origOp.filter().getType().cast<vpux::NDTypeInterface>();
    const auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        const auto inputTiles = origOp.backInferTileInfo(outputTile, log).tiles;

        VPUX_THROW_UNLESS(inputTiles.size() > 1, "Missed tile information. Got {0} tiles info, must be at least 2",
                          inputTiles.size());
        const auto& inputTile = inputTiles[0];
        const auto& filterTile = inputTiles[1];

        const auto inputTileType = inputType.extractDenseTile(inputTile.offsets, inputTile.shape);
        const auto filterTileType = filterType.extractDenseTile(filterTile.offsets, filterTile.shape);
        const auto outputTileType = outputType.extractDenseTile(outputTile.offsets, outputTile.shape);

        if (origOp->hasAttr(VPU::multiClusterStrategy)) {
            auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
            VPUX_THROW_WHEN(nceOp == nullptr, "Op {0} has multiClusterStrategy but is not an NCEOp", origOp->getLoc());

            auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
            VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                            origOp->getLoc());

            auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTileType.getShape()[Dims4D::Act::C],
                                                          clusteredOp.getMultiClusterStrategyAttr().getValue());
            return origOp.fitIntoCMX(VPU::getDistributedActivationTypeFromOp(clusteredOp, inputTileType, numClusters),
                                     VPU::getDistributedFilterTypeFromOp(nceOp, filterTileType, numClusters),
                                     VPU::getDistributedOutputTypeFromOp(clusteredOp, outputTileType, numClusters));
        }
        return origOp.fitIntoCMX(inputTileType, filterTileType, outputTileType);
    });
}

bool isSupportedIsolatedTiling(VPU::GroupConvolutionOp origOp, const OutputTiling& tiles, Logger log) {
    const auto inputType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto filterType = origOp.filter().getType().cast<vpux::NDTypeInterface>();
    const auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        const auto origInputShape = getShape(origOp.input());
        const auto origFilterShape = getShape(origOp.filter());
        const auto origBiasShape = origOp.bias() != nullptr ? getShape(origOp.bias()) : ShapeRef();
        const auto origPadding = PadInfo(origOp.pads_begin(), origOp.pads_end());

        const auto inputTiling = backInferGroupConvTile(outputTile, origInputShape, origFilterShape, origBiasShape,
                                                        origOp.strides(), origPadding);

        const auto& tileConf = inputTiling.tiles;
        VPUX_THROW_UNLESS(tileConf.size() > 1, "Missed tile information. Got {0} tiles info, must be at least 2",
                          tileConf.size());
        const auto& inputTile = tileConf[0];
        const auto& filterTile = tileConf[1];

        const auto inputTileType = inputType.extractDenseTile(inputTile.offsets, inputTile.shape);
        const auto filterTileType = filterType.extractDenseTile(filterTile.offsets, filterTile.shape);
        const auto outputTileType = outputType.extractDenseTile(outputTile.offsets, outputTile.shape);

        return mlir::succeeded(VPUIP::NCEInvariant::verifyGroupConvCMX(
                origOp->getLoc(), origOp->getParentOfType<mlir::ModuleOp>(), inputTileType, filterTileType,
                outputTileType, origOp.strides(), log));
    });
}

bool isSupportedIsolatedTiling(VPU::NCEDepthConvolutionOp origOp, const OutputTiling& tiles, Logger log) {
    const auto inputType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto filterType = origOp.filter().getType().cast<vpux::NDTypeInterface>();
    const auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        const auto inputTiles = origOp.backInferTileInfo(outputTile, log).tiles;

        VPUX_THROW_UNLESS(inputTiles.size() > 1, "Missed tile information. Got {0} tiles info, must be at least 2",
                          inputTiles.size());
        const auto& inputTile = inputTiles[0];
        const auto& filterTile = inputTiles[1];

        const auto inputTileType = inputType.extractDenseTile(inputTile.offsets, inputTile.shape);
        const auto filterTileType = filterType.extractDenseTile(filterTile.offsets, filterTile.shape);
        const auto outputTileType = outputType.extractDenseTile(outputTile.offsets, outputTile.shape);

        if (origOp->hasAttr(VPU::multiClusterStrategy)) {
            auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
            VPUX_THROW_WHEN(nceOp == nullptr, "Op {0} has multiClusterStrategy but is not an NCEOp", origOp->getLoc());
            auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
            VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                            origOp->getLoc());

            auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTileType.getShape()[Dims4D::Act::C],
                                                          clusteredOp.getMultiClusterStrategyAttr().getValue());
            return origOp.fitIntoCMX(VPU::getDistributedActivationTypeFromOp(clusteredOp, inputTileType, numClusters),
                                     VPU::getDistributedFilterTypeFromOp(nceOp, filterTileType, numClusters),
                                     VPU::getDistributedOutputTypeFromOp(clusteredOp, outputTileType, numClusters));
        }
        return origOp.fitIntoCMX(inputTileType, filterTileType, outputTileType);
    });
}

bool isSupportedIsolatedTiling(VPU::MaxPoolOp origOp, const OutputTiling& tiles, Logger log) {
    const auto inputType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        const auto origInputShape = getShape(origOp.input());
        const auto origPadding = PadInfo(origOp.pads_begin(), origOp.pads_end());

        const auto inputTiling =
                backInferPoolTile(outputTile, origInputShape, origOp.kernel_size(), origOp.strides(), origPadding);

        const auto& tileConf = inputTiling.tiles;
        VPUX_THROW_UNLESS(!tileConf.empty(), "Got empty tile information");
        const auto& inputTile = tileConf[0];

        const auto inputTileType = inputType.extractDenseTile(inputTile.offsets, inputTile.shape);
        const auto outputTileType = outputType.extractDenseTile(outputTile.offsets, outputTile.shape);

        return mlir::succeeded(VPUIP::NCEInvariant::verifyPoolCMX(
                origOp->getLoc(), origOp->getParentOfType<mlir::ModuleOp>(), inputTileType, outputTileType,
                origOp.kernel_size(), origOp.strides(), log));
    });
}

bool isSupportedIsolatedTiling(VPU::NCEMaxPoolOp origOp, const OutputTiling& tiles, Logger log) {
    const auto inputType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        const auto inputTiles = origOp.backInferTileInfo(outputTile, log).tiles;

        VPUX_THROW_UNLESS(!inputTiles.empty(), "Got empty tile information");
        const auto& inputTile = inputTiles[0];

        const auto inputTileType = inputType.extractDenseTile(inputTile.offsets, inputTile.shape);
        const auto outputTileType = outputType.extractDenseTile(outputTile.offsets, outputTile.shape);

        if (origOp->hasAttr(VPU::multiClusterStrategy)) {
            auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
            VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                            origOp->getLoc());

            auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTileType.getShape()[Dims4D::Act::C],
                                                          clusteredOp.getMultiClusterStrategyAttr().getValue());
            return origOp.fitIntoCMX(VPU::getDistributedActivationTypeFromOp(clusteredOp, inputTileType, numClusters),
                                     VPU::getDistributedOutputTypeFromOp(clusteredOp, outputTileType, numClusters));
        }
        return origOp.fitIntoCMX(inputTileType, outputTileType);
    });
}

bool isSupportedIsolatedTiling(VPU::NCEAveragePoolOp origOp, const OutputTiling& tiles, Logger log) {
    const auto inputType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        const auto inputTiles = origOp.backInferTileInfo(outputTile, log).tiles;

        VPUX_THROW_UNLESS(!inputTiles.empty(), "Got empty tile information");
        const auto& inputTile = inputTiles[0];

        const auto inputTileType = inputType.extractDenseTile(inputTile.offsets, inputTile.shape);
        const auto outputTileType = outputType.extractDenseTile(outputTile.offsets, outputTile.shape);

        if (origOp->hasAttr(VPU::multiClusterStrategy)) {
            auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
            VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                            origOp->getLoc());
            auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTileType.getShape()[Dims4D::Act::C],
                                                          clusteredOp.getMultiClusterStrategyAttr().getValue());
            return origOp.fitIntoCMX(VPU::getDistributedActivationTypeFromOp(clusteredOp, inputTileType, numClusters),
                                     VPU::getDistributedOutputTypeFromOp(clusteredOp, outputTileType, numClusters));
        }
        return origOp.fitIntoCMX(inputTileType, outputTileType);
    });
}

bool isSupportedIsolatedTilingEltwise(mlir::Operation* origOp, const OutputTiling& tiles, Logger log) {
    const auto input1Type = origOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto input2Type = origOp->getOperand(1).getType().cast<vpux::NDTypeInterface>();
    const auto outputType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    return llvm::all_of(tiles, [&](const TileInfo& tile) {
        const auto input1TileType = input1Type.extractDenseTile(tile.offsets, tile.shape);
        const auto input2TileType = input2Type.extractDenseTile(tile.offsets, tile.shape);
        const auto outputTileType = outputType.extractDenseTile(tile.offsets, tile.shape);

        auto isInplace = false;
        if (auto nceEltwiseOp = mlir::dyn_cast<VPU::NCEEltwiseOp>(origOp)) {
            isInplace = nceEltwiseOp.is_inplace().getValueOr(false);
        }

        if (origOp->hasAttr(VPU::multiClusterStrategy)) {
            auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp);
            VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not a ClusteredOp",
                            origOp->getLoc());
            auto module = clusteredOp->getParentOfType<mlir::ModuleOp>();
            auto numClusters = VPU::getOptimalNumClusters(
                    clusteredOp, outputTileType.getShape()[Dims4D::Act::C],
                    clusteredOp->getAttr(VPU::multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue());

            return mlir::succeeded(VPUIP::NCEInvariant::verifyEltwiseCMX(
                    origOp->getLoc(), module, isInplace,
                    VPU::getDistributedActivationTypeFromOp(clusteredOp, input1TileType, numClusters),
                    VPU::getDistributedActivationTypeFromOp(clusteredOp, input2TileType, numClusters),
                    VPU::getDistributedOutputTypeFromOp(clusteredOp, outputTileType, numClusters)));
        }
        return mlir::succeeded(
                VPUIP::NCEInvariant::verifyEltwiseCMX(origOp->getLoc(), origOp->getParentOfType<mlir::ModuleOp>(),
                                                      isInplace, input1TileType, input2TileType, outputTileType, log));
    });
}

bool isSupportedIsolatedTilingSwInterface(VPU::SWOpInterface origOp, const OutputTiling& tiles, Logger log) {
    log.trace("isSupportedIsolatedTilingSwInterface OpName: {0}", origOp->getName());
    const auto inputType = origOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>();

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        const auto inputTileType = inputType.extractDenseTile(outputTile.offsets, outputTile.shape);
        const auto outputTileType = outputType.extractDenseTile(outputTile.offsets, outputTile.shape);

        if (origOp->hasAttr(VPU::multiClusterStrategy)) {
            auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
            VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                            origOp->getLoc());
            auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTileType.getShape()[Dims4D::Act::C],
                                                          clusteredOp.getMultiClusterStrategyAttr().getValue());
            return origOp.fitIntoCMX(VPU::getDistributedActivationTypeFromOp(clusteredOp, inputTileType, numClusters),
                                     VPU::getDistributedOutputTypeFromOp(clusteredOp, outputTileType, numClusters));
        }
        return origOp.fitIntoCMX(inputTileType, outputTileType);
    });
}

bool isSupportedIsolatedTilingGeneric(mlir::Operation* origOp, const OutputTiling& tiles, Logger log) {
    const auto operands = origOp->getOperands();
    const auto results = origOp->getResults();

    auto tilingOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(origOp);
    VPUX_THROW_UNLESS(tilingOp != nullptr, "Not a tileable operation {0}", origOp->getName());
    const auto cmxAvailableBytes = vpux::VPU::getTotalCMXSize(origOp).to<Byte>().count();

    int64_t outputByteSize = 0;
    log.trace("isSupportedIsolatedTilingGeneric OpName: {0}", origOp->getName());
    auto outshape = results[0].getType().cast<vpux::NDTypeInterface>().getShape();
    // TODO: We should proper implement multiple outputs Op support when or with E#59993 implementation.
    // Now it is supported and assumed just opperation with same tiling and same shape for multiple outputs.
    for (auto p : results) {
        const auto outputType = p.getType().cast<vpux::NDTypeInterface>();
        outputByteSize += outputType.getElemTypeSize().to<Byte>().count();
        VPUX_THROW_UNLESS(outshape == outputType.getShape(),
                          "Expected operation with same shape of results , got {0} and {1} results", outshape,
                          outputType.getShape());
    }

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        auto inputTiles = tilingOp.backInferTileInfo(outputTile, log);
        if (inputTiles.tiles.size() < 1) {
            log.trace("No input tiles for {0}", origOp->getLoc());
            return false;
        }
        const auto& inTiles = inputTiles.tiles;
        const auto outputTileSizeBytes = outputTile.shape.totalSize() * outputByteSize;
        log.trace("outputTileSizeBytes: {0}", outputTileSizeBytes);
        auto requiredCMX = outputTileSizeBytes;
        for (auto p : inTiles | indexed) {
            const auto inT = p.value();
            const auto index = p.index();
            const auto inputType = operands[index].getType().cast<vpux::NDTypeInterface>();
            const auto inputByteSize = inputType.getElemTypeSize().to<Byte>().count();
            const auto inputTileSizeBytes = inT.shape.totalSize() * inputByteSize;
            requiredCMX += inputTileSizeBytes;
        }
        if (requiredCMX > cmxAvailableBytes) {
            log.trace(
                    "Tile does not fit into CMX for op {0}. Input tile[0] {1}, output tile {2}, required CMX size {3}, "
                    "max available MX: {4}",
                    origOp->getLoc(), inTiles[0].shape, outputTile.shape, requiredCMX, cmxAvailableBytes);
            return false;
        }
        log.trace("Op {0} out tiling probe valid: {1} - input tile on 0 pos: {2}", origOp->getLoc(), outputTile,
                  inTiles[0]);
        return true;
    });
}

bool isSupportedIsolatedTiling(VPU::NCEPermuteQuantizeOp origOp, const OutputTiling& tiles, Logger log) {
    const auto inputType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        const auto inputTiles = origOp.backInferTileInfo(outputTile, log).tiles;

        VPUX_THROW_UNLESS(inputTiles.size() > 0, "Missed tile information. Got {0} tiles info, must be at least 1",
                          inputTiles.size());
        const auto& inputTile = inputTiles[0];
        const auto inputTileType = inputType.extractDenseTile(inputTile.offsets, inputTile.shape);
        const auto outputTileType = outputType.extractDenseTile(outputTile.offsets, outputTile.shape);
        if (origOp->hasAttr(VPU::multiClusterStrategy)) {
            auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
            VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                            origOp->getLoc());
            // isSupportedIsolatedTiling operates with reshape inputs.
            // For example, 1x3x16x32 becomes 1x32x3x16.
            // Network input is tiled by height, but the number of clusters must be calculated based on width.
            auto numClusters = VPU::getOptimalNumClusters(
                    clusteredOp, outputTileType.getShape()[Dims4D::Act::W],
                    clusteredOp->getAttr(VPU::multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue());
            return origOp.fitIntoCMX(VPU::getDistributedActivationTypeFromOp(clusteredOp, inputTileType, numClusters),
                                     VPU::getDistributedOutputTypeFromOp(clusteredOp, outputTileType, numClusters));
        }
        return origOp.fitIntoCMX(inputTileType, outputTileType);
    });
}

bool isSupportedIsolatedTilingSwLayer(mlir::Operation* origOp, const OutputTiling& tiles, Logger log) {
    return llvm::TypeSwitch<mlir::Operation*, bool>(origOp)
            .Case<VPU::ConvolutionOp>([&](VPU::ConvolutionOp op) {
                return isSupportedIsolatedTiling(op, tiles, log);
            })
            .Case<VPU::GroupConvolutionOp>([&](VPU::GroupConvolutionOp op) {
                return isSupportedIsolatedTiling(op, tiles, log);
            })
            .Case<VPU::MaxPoolOp>([&](VPU::MaxPoolOp op) {
                return isSupportedIsolatedTiling(op, tiles, log);
            })
            .Case<VPU::AddOp, VPU::MultiplyOp, VPU::SubtractOp, VPU::AndOp>([&](mlir::Operation* op) {
                return isSupportedIsolatedTilingEltwise(op, tiles, log);
            })
            .Case<VPU::SWOpInterface>([&](VPU::SWOpInterface swOp) {
                return isSupportedIsolatedTilingSwInterface(swOp, tiles, log);
            })
            .Default([&](mlir::Operation* op) -> bool {
                return isSupportedIsolatedTilingGeneric(op, tiles, log);
            });
}

SmallVector<Dim> getTileDims(ShapeRef tileAxis) {
    SmallVector<Dim> tileDims;
    for (unsigned i = 0; i < tileAxis.size(); i++) {
        if (tileAxis[Dim(i)] > 1)
            tileDims.emplace_back(Dim(i));
    }
    return tileDims;
}

bool isLastTileBiggest(mlir::Operation* op, ShapeRef tileAxis, ShapeRef outputShape, Dim tileDim) {
    auto tileResult = fillDividedTiles(op, tileAxis, outputShape);
    auto lastTile = tileResult.end() - 1;
    auto firstTile = tileResult.begin();
    return lastTile->shape[tileDim] > firstTile->shape[tileDim];
}

bool isDivisibleTile(mlir::Operation* op, ShapeRef tileAxis, Dim tileDim, int64_t kernelSize) {
    int64_t minChannelSize = 1;
    if (auto channelsInfo = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
        minChannelSize = channelsInfo.getOutputChannelAlignment();
    }
    auto outputShape = getShape(op->getResult(0));
    if (tileDim == Dims4D::Act::C) {
        return (outputShape[tileDim] / tileAxis[tileDim] >= minChannelSize) &&
               (outputShape[tileDim] % tileAxis[tileDim] == 0) &&
               ((outputShape[tileDim] / tileAxis[tileDim]) % minChannelSize == 0);
    } else {
        return outputShape[tileDim] / tileAxis[tileDim] >= kernelSize;
    }
}

bool checkPrefetchMem(mlir::Operation* op, const OutputTiling& tiles, Logger log) {
    auto parentOp = VPU::getParentTargetOp(op);
    const auto parentShape = getShape(parentOp->getResult(0));
    return mlir::succeeded(
            vpux::VPUIP::NCEInvariant::verifyPrefetchCMX(op, tiles, parentOp, {TileInfo(parentShape)}, log));
}

bool isSupportedPrefetchTiling(VPU::NCEConvolutionOp origOp, const OutputTiling& tiles, Logger log,
                               TilingMode tilingMode) {
    auto outputShape = getShape(origOp.output());
    auto tileAxis = tiles.begin()->axis;
    auto tileDims = getTileDims(tileAxis);

    auto isMemPrefetchable = [&]() -> bool {
        if (tilingMode == vpux::TilingMode::PIPELINING) {
            return vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(origOp, tiles, log).succeeded();
        }
        // Pattern prefetch
        return checkPrefetchMem(origOp.getOperation(), tiles, log);
    };

    // neutral tiling check
    if (tileDims.size() == 0 && tilingMode == vpux::TilingMode::PREFETCHING) {
        return isMemPrefetchable();
    }

    // Prefetch tiling is only triggered when the isolated tiling is not nested
    if (tileDims.size() != 1) {
        return false;
    }
    auto tileDim = tileDims[0];
    const auto rawFilterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShape()));
    return isDivisibleTile(origOp.getOperation(), tileAxis, tileDim, rawFilterShape[tileDim]) && isMemPrefetchable() &&
           !isLastTileBiggest(origOp.getOperation(), tileAxis, outputShape, tileDim);
}

bool isSupportedPrefetchTiling(VPU::NCEDepthConvolutionOp origOp, const OutputTiling& tiles, Logger log,
                               TilingMode tilingMode) {
    auto outputShape = getShape(origOp.output());
    auto tileAxis = tiles.begin()->axis;
    auto tileDims = getTileDims(tileAxis);

    auto isMemPrefetchable = [&]() -> bool {
        if (tilingMode == vpux::TilingMode::PIPELINING) {
            return vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(origOp, tiles, log).succeeded();
        }
        // Pattern prefetch
        return checkPrefetchMem(origOp.getOperation(), tiles, log);
    };

    // neutral tiling check
    if (tileDims.size() == 0 && tilingMode == vpux::TilingMode::PREFETCHING) {
        return isMemPrefetchable();
    }

    // Prefetch tiling is only triggered when the isolated tiling is not nested
    if (tileDims.size() != 1) {
        return false;
    }
    auto tileDim = tileDims[0];
    const auto rawFilterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShape()));

    return isDivisibleTile(origOp.getOperation(), tileAxis, tileDim, rawFilterShape[tileDim]) && isMemPrefetchable() &&
           !isLastTileBiggest(origOp.getOperation(), tileAxis, outputShape, tileDim);
}

bool isSupportedPrefetchTiling(VPU::NCEMaxPoolOp origOp, const OutputTiling& tiles, Logger log, TilingMode tilingMode) {
    auto tileAxis = tiles.begin()->axis;
    auto tileDims = getTileDims(tileAxis);

    auto isMemPrefetchable = [&]() -> bool {
        if (tilingMode == vpux::TilingMode::PIPELINING) {
            return vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(origOp, tiles, log).succeeded();
        }
        // Pattern prefetch
        return checkPrefetchMem(origOp.getOperation(), tiles, log);
    };

    // neutral tiling check
    if (tileDims.size() == 0 && tilingMode == vpux::TilingMode::PREFETCHING) {
        return isMemPrefetchable();
    }

    // Prefetch tiling is only triggered when the isolated tiling is not nested
    if (tileDims.size() != 1) {
        return false;
    }
    auto tileDim = tileDims[0];
    auto outputShape = getShape(origOp.output());

    size_t realKernelIndex = tileDim == Dims4D::Act::H ? 0 : 1;
    return isDivisibleTile(origOp.getOperation(), tileAxis, tileDim,
                           parseIntArrayAttr<int64_t>(origOp.kernel_size())[realKernelIndex]) &&
           isMemPrefetchable() && !isLastTileBiggest(origOp.getOperation(), tileAxis, outputShape, tileDim);
}

bool isSupportedPrefetchTiling(VPU::NCEAveragePoolOp origOp, const OutputTiling& tiles, Logger log,
                               TilingMode tilingMode) {
    auto tileAxis = tiles.begin()->axis;
    auto tileDims = getTileDims(tileAxis);

    auto isMemPrefetchable = [&]() -> bool {
        if (tilingMode == vpux::TilingMode::PIPELINING) {
            return vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(origOp, tiles, log).succeeded();
        }
        // Pattern prefetch
        return checkPrefetchMem(origOp.getOperation(), tiles, log);
    };

    // neutral tiling check
    if (tileDims.size() == 0 && tilingMode == vpux::TilingMode::PREFETCHING) {
        return isMemPrefetchable();
    }

    // Prefetch tiling is only triggered when the isolated tiling is not nested
    if (tileDims.size() != 1) {
        return false;
    }
    auto tileDim = tileDims[0];
    auto outputShape = getShape(origOp.output());

    size_t realKernelIndex = tileDim == Dims4D::Act::H ? 0 : 1;
    return isDivisibleTile(origOp.getOperation(), tileAxis, tileDim,
                           parseIntArrayAttr<int64_t>(origOp.kernel_size())[realKernelIndex]) &&
           isMemPrefetchable() && !isLastTileBiggest(origOp.getOperation(), tileAxis, outputShape, tileDim);
}

bool isSupportedPrefetchTiling(VPU::NCEPermuteQuantizeOp origOp, const OutputTiling& /*tiles*/, Logger log,
                               TilingMode /*tilingMode*/) {
    // NCE.PermuteQuantize will be lowered to eltwise add, same rules are applied.
    // The DPU time of any eltwise operation is too short, it's not worth prefetching.
    log.trace("Op {0} does not support prefetch tiling", origOp->getLoc());
    return false;
}

template <class MainOpType>
class NCETilingInfoOpModel final :
        public VPU::TilingInfoOpInterface::ExternalModel<NCETilingInfoOpModel<MainOpType>, MainOpType> {
public:
    bool isSupportedTiling(mlir::Operation* origOp, const OutputTiling& tiles, TilingMode tilingMode,
                           Logger log) const {
        if (!isSupportedByNCE(mlir::cast<MainOpType>(origOp), log)) {
            return true;
        }
        switch (tilingMode) {
        case vpux::TilingMode::ISOLATED:
            return isSupportedIsolatedTiling(mlir::cast<MainOpType>(origOp), tiles, log);
        case vpux::TilingMode::PIPELINING:
        case vpux::TilingMode::PREFETCHING:
            return isSupportedPrefetchTiling(mlir::cast<MainOpType>(origOp), tiles, log, tilingMode);
        default:
            VPUX_THROW("Unknown tiling mode: '{0}'.", getTilingModeStr(tilingMode));
        }
    }

private:
    static bool isSupportedByNCE(MainOpType op, Logger log) {
        if (VPU::getCompilationMode(op) == VPU::CompilationMode::ReferenceSW) {
            return false;
        }

        const bool kernelVerified = VPUIP::NCEInvariant::verifyKernel(op, log).succeeded();
        const bool channelsVerified = VPUIP::NCEInvariant::verifyChannels(op, log).succeeded();
        return kernelVerified && channelsVerified;
    }
};

template <class MainOpType>
class NCEEltwiseTilingInfoOpModel final :
        public VPU::TilingInfoOpInterface::ExternalModel<NCEEltwiseTilingInfoOpModel<MainOpType>, MainOpType> {
public:
    bool isSupportedTiling(mlir::Operation* origOp, const OutputTiling& tiles, TilingMode tilingMode,
                           Logger log) const {
        if (!isSupportedByNCE(mlir::cast<MainOpType>(origOp), log)) {
            return true;
        }

        switch (tilingMode) {
        case TilingMode::ISOLATED:
            return ::isSupportedIsolatedTilingEltwise(origOp, tiles, log);
        case TilingMode::PIPELINING:
        case TilingMode::PREFETCHING:
            // The DPU time of eltwise operations are too short to worth prefetching.
            return false;
        default:
            VPUX_THROW("Unknown tiling mode. ISOLATED, PIPELINING and PREFETCHING are supported.");
        }
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
class SwLayerTilingInfoOpModel final :
        public VPU::TilingInfoOpInterface::ExternalModel<SwLayerTilingInfoOpModel<MainOpType>, MainOpType> {
public:
    bool isSupportedTiling(mlir::Operation* origOp, const OutputTiling& tiles, TilingMode tilingMode,
                           Logger log) const {
        switch (tilingMode) {
        case vpux::TilingMode::ISOLATED:
            return ::isSupportedIsolatedTilingSwLayer(origOp, tiles, log);
        case vpux::TilingMode::PIPELINING:
        case vpux::TilingMode::PREFETCHING:
            return false;
        default:
            VPUX_THROW("Unknown tiling mode: '{0}'.", getTilingModeStr(tilingMode));
        }
    }
};

//
// AsyncLayerOpModel
//

class AsyncLayerOpModelForDMA final : public VPUIP::AsyncLayerOpInterface::FallbackModel<AsyncLayerOpModelForDMA> {
public:
    IndexedSymbolAttr getExecutor(mlir::Operation* origOp) const {
        return VPUIP::getExecutorAttr(origOp, VPU::ExecutorKind::DMA_NN);
    }
};

class AsyncLayerOpModelForSW final : public VPUIP::AsyncLayerOpInterface::FallbackModel<AsyncLayerOpModelForSW> {
public:
    IndexedSymbolAttr getExecutor(mlir::Operation* origOp) const {
        return VPUIP::getExecutorAttr(origOp, VPU::ExecutorKind::SHAVE_UPA);
    }
};

//
// SoftwareLayerOpModel
//

class SoftwareLayerOpModel final : public VPUIP::SoftwareLayerOpInterface::FallbackModel<SoftwareLayerOpModel> {
public:
    VPUIP::KernelInfo getKernelInfo(mlir::Operation* origOp) const {
        return VPUIP::SwKernelOp::getKernelInfo(origOp);
    }
};

//
// DummySoftwareLayerOpModel
//

class DummySoftwareLayerOpModel final :
        public VPUIP::SoftwareLayerOpInterface::FallbackModel<DummySoftwareLayerOpModel> {
public:
    VPUIP::KernelInfo getKernelInfo(mlir::Operation* /*origOp*/) const {
        return VPUIP::SwKernelOp::getDummyKernelInfo();
    }
};

//
// redirectOpInterfacesForIE
//

template <template <class, class> class OpModelForHW, template <class> class OpModelForSW>
void redirectOpInterfacesForIE(mlir::DialectRegistry& registry) {
    registry.addOpInterface<IE::ConvolutionOp, OpModelForHW<IE::ConvolutionOp, VPU::ConvolutionOp>>();
    registry.addOpInterface<IE::GroupConvolutionOp, OpModelForHW<IE::GroupConvolutionOp, VPU::ConvolutionOp>>();
    registry.addOpInterface<IE::MaxPoolOp, OpModelForHW<IE::MaxPoolOp, VPU::AvgPoolOp>>();
    registry.addOpInterface<IE::AddOp, OpModelForHW<IE::AddOp, VPU::AddOp>>();
    registry.addOpInterface<IE::MultiplyOp, OpModelForHW<IE::MultiplyOp, VPU::MultiplyOp>>();
    registry.addOpInterface<IE::SubtractOp, OpModelForHW<IE::SubtractOp, VPU::SubtractOp>>();
    registry.addOpInterface<IE::AndOp, OpModelForHW<IE::AndOp, VPU::AndOp>>();
}

//
// redirectOpInterfacesForVPUIP
//

template <class OpModelForDMA, class OpModelForSW>
void redirectOpInterfacesForVPUIP(mlir::DialectRegistry& registry) {
    registry.addOpInterface<VPUIP::CopyOp, OpModelForDMA>();
    registry.addOpInterface<VPUIP::TimestampOp, OpModelForDMA>();
    registry.addOpInterface<VPUIP::DepthToSpaceDMAOp, OpModelForDMA>();
    registry.addOpInterface<VPUIP::PermuteDMAOp, OpModelForDMA>();
    registry.addOpInterface<VPUIP::ExpandDMAOp, OpModelForDMA>();
    registry.addOpInterface<VPUIP::ExpandOp, OpModelForDMA>();

    registry.addOpInterface<VPUIP::ConvertUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::SoftMaxUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::PoolingUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::AdaptiveAvgPoolUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::AdaptiveMaxPoolUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::ReLUUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::SigmoidUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::SignUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::ClampUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::EluUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::HSwishUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::MishUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::ErfUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::BroadcastUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::FloorUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::RoundUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::TanUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::TanhUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::SinUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::CosUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::SqrtUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::SinhUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::SeluUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::CoshUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::AsinhUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::AcoshUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::AtanhUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::LogUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::GeluUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::BucketizeUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::QuantCastUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::FakeQuantizeUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::GatherUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::GatherNDUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::ScatterUpdateUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::YuvToRgbUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::GatherElementsUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::ScatterNDUpdateUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::PReluUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::LeakyReluUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::SwishUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::GRNUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::NormUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::ReduceUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::PerAxisTileUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::NegativeUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::ROIPoolingUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::PSROIPoolingUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::ROIAlignUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::ProposalUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::FullyConnectedUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::DetectionOutputUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::ScaleShiftUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::CTCGreedyDecoderUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::CTCGreedyDecoderSeqLenUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::PadUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::ExpUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::InterpolateUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::LSTMCellUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::StridedSliceUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::RegionYoloUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::ReorgYoloUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::MVNUPAOp, OpModelForSW>();
    registry.addOpInterface<VPU::MVNOp, SwLayerTilingInfoOpModel<VPU::MVNOp>>();
    registry.addOpInterface<VPUIP::LSTMSequenceUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::PermuteUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::CeilingUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::NormalizeIEUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::CumSumUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::SelectUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::DepthToSpaceUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::ReverseSequenceUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::UpsamplingUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::SoftPlusUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::TopKUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::LogicalNotUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::SpaceToDepthUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::ExtractImagePatchesUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::AbsUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::HSigmoidUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::RollUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::AtanUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::AsinUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::AcosUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::HardSigmoidUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::EmbeddingBagOffsetsSumUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::EmbeddingSegmentsSumUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::DeformablePSROIPoolingUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::NonMaxSuppressionUPAOp, OpModelForSW>();
    registry.addOpInterface<VPUIP::LSTMCellUPAOp, OpModelForSW>();
}

}  // namespace

//
// setupExtraInterfaces
//

void vpux::VPUIP::VPUIPDialect::setupExtraInterfaces(mlir::DialectRegistry& registry) {
    registry.addOpInterface<IE::ConvolutionOp, LayerWithPostOpModel<IE::ConvolutionOp>>();
    registry.addOpInterface<IE::DeconvolutionOp, LayerWithPostOpModel<IE::DeconvolutionOp>>();
    registry.addOpInterface<IE::GroupConvolutionOp, LayerWithPostOpModel<IE::GroupConvolutionOp>>();
    registry.addOpInterface<IE::MaxPoolOp, LayerWithPostOpModel<IE::MaxPoolOp>>();
    registry.addOpInterface<IE::AvgPoolOp, LayerWithPostOpModel<IE::AvgPoolOp>>();
    registry.addOpInterface<IE::AddOp, LayerWithPostOpModel<IE::AddOp>>();
    registry.addOpInterface<IE::MultiplyOp, LayerWithPostOpModel<IE::MultiplyOp>>();
    registry.addOpInterface<IE::SubtractOp, LayerWithPostOpModel<IE::SubtractOp>>();
    registry.addOpInterface<IE::AndOp, LayerWithPostOpModel<IE::AndOp>>();

    registry.addOpInterface<IE::ConvolutionOp, AlignedChannelsOpModel<IE::ConvolutionOp>>();
    registry.addOpInterface<IE::GroupConvolutionOp, AlignedChannelsOpModel<IE::GroupConvolutionOp>>();
    registry.addOpInterface<IE::MaxPoolOp, AlignedChannelsOpModel<IE::MaxPoolOp>>();
    registry.addOpInterface<IE::AvgPoolOp, AlignedChannelsOpModel<IE::AvgPoolOp>>();
    registry.addOpInterface<IE::AddOp, AlignedChannelsOpModel<IE::AddOp>>();
    registry.addOpInterface<IE::MultiplyOp, AlignedChannelsOpModel<IE::MultiplyOp>>();
    registry.addOpInterface<IE::SubtractOp, AlignedChannelsOpModel<IE::SubtractOp>>();
    registry.addOpInterface<IE::AndOp, AlignedChannelsOpModel<IE::AndOp>>();

    registry.addOpInterface<VPU::NCEConvolutionOp, NCETilingInfoOpModel<VPU::NCEConvolutionOp>>();
    registry.addOpInterface<VPU::NCEDepthConvolutionOp, NCETilingInfoOpModel<VPU::NCEDepthConvolutionOp>>();
    registry.addOpInterface<VPU::NCEMaxPoolOp, NCETilingInfoOpModel<VPU::NCEMaxPoolOp>>();
    registry.addOpInterface<VPU::NCEAveragePoolOp, NCETilingInfoOpModel<VPU::NCEAveragePoolOp>>();
    registry.addOpInterface<VPU::NCEEltwiseOp, NCEEltwiseTilingInfoOpModel<VPU::NCEEltwiseOp>>();
    registry.addOpInterface<VPU::NCEPermuteQuantizeOp, NCETilingInfoOpModel<VPU::NCEPermuteQuantizeOp>>();

    registry.addOpInterface<VPU::ConvolutionOp, SwLayerTilingInfoOpModel<VPU::ConvolutionOp>>();
    registry.addOpInterface<VPU::GroupConvolutionOp, SwLayerTilingInfoOpModel<VPU::GroupConvolutionOp>>();
    registry.addOpInterface<VPU::MaxPoolOp, SwLayerTilingInfoOpModel<VPU::MaxPoolOp>>();
    registry.addOpInterface<VPU::AddOp, SwLayerTilingInfoOpModel<VPU::AddOp>>();
    registry.addOpInterface<VPU::MultiplyOp, SwLayerTilingInfoOpModel<VPU::MultiplyOp>>();
    registry.addOpInterface<VPU::SubtractOp, SwLayerTilingInfoOpModel<VPU::SubtractOp>>();
    registry.addOpInterface<VPU::AndOp, SwLayerTilingInfoOpModel<VPU::AndOp>>();
    registry.addOpInterface<VPU::InterpolateOp, SwLayerTilingInfoOpModel<VPU::InterpolateOp>>();
    registry.addOpInterface<VPU::GatherOp, SwLayerTilingInfoOpModel<VPU::GatherOp>>();
    registry.addOpInterface<VPU::GatherNDOp, SwLayerTilingInfoOpModel<VPU::GatherNDOp>>();
    registry.addOpInterface<VPU::ConvertOp, SwLayerTilingInfoOpModel<VPU::ConvertOp>>();
    registry.addOpInterface<VPU::SigmoidOp, SwLayerTilingInfoOpModel<VPU::SigmoidOp>>();
    registry.addOpInterface<VPU::HSwishOp, SwLayerTilingInfoOpModel<VPU::HSwishOp>>();
    registry.addOpInterface<VPU::HSigmoidOp, SwLayerTilingInfoOpModel<VPU::HSigmoidOp>>();
    registry.addOpInterface<VPU::LeakyReluOp, SwLayerTilingInfoOpModel<VPU::LeakyReluOp>>();
    registry.addOpInterface<VPU::PReluOp, SwLayerTilingInfoOpModel<VPU::PReluOp>>();
    registry.addOpInterface<VPU::EluOp, SwLayerTilingInfoOpModel<VPU::EluOp>>();
    registry.addOpInterface<VPU::ClampOp, SwLayerTilingInfoOpModel<VPU::ClampOp>>();
    registry.addOpInterface<VPU::ReLUOp, SwLayerTilingInfoOpModel<VPU::ReLUOp>>();
    registry.addOpInterface<VPU::SqrtOp, SwLayerTilingInfoOpModel<VPU::SqrtOp>>();
    registry.addOpInterface<VPU::ExpOp, SwLayerTilingInfoOpModel<VPU::ExpOp>>();
    registry.addOpInterface<VPU::TanhOp, SwLayerTilingInfoOpModel<VPU::TanhOp>>();
    registry.addOpInterface<VPU::DivideOp, SwLayerTilingInfoOpModel<VPU::DivideOp>>();
    registry.addOpInterface<VPU::FloorOp, SwLayerTilingInfoOpModel<VPU::FloorOp>>();
    registry.addOpInterface<VPU::MemPermuteOp, SwLayerTilingInfoOpModel<VPU::MemPermuteOp>>();
    registry.addOpInterface<VPU::AvgPoolOp, SwLayerTilingInfoOpModel<VPU::AvgPoolOp>>();
    registry.addOpInterface<VPU::PermuteQuantizeOp, SwLayerTilingInfoOpModel<VPU::PermuteQuantizeOp>>();
    registry.addOpInterface<VPU::LogOp, SwLayerTilingInfoOpModel<VPU::LogOp>>();
    registry.addOpInterface<VPU::PowerOp, SwLayerTilingInfoOpModel<VPU::PowerOp>>();
    registry.addOpInterface<VPU::FloorModOp, SwLayerTilingInfoOpModel<VPU::FloorModOp>>();
    registry.addOpInterface<VPU::EqualOp, SwLayerTilingInfoOpModel<VPU::EqualOp>>();
    registry.addOpInterface<VPU::LessOp, SwLayerTilingInfoOpModel<VPU::LessOp>>();
    registry.addOpInterface<VPU::LessEqualOp, SwLayerTilingInfoOpModel<VPU::LessEqualOp>>();
    registry.addOpInterface<VPU::NotEqualOp, SwLayerTilingInfoOpModel<VPU::NotEqualOp>>();
    registry.addOpInterface<VPU::GreaterOp, SwLayerTilingInfoOpModel<VPU::GreaterOp>>();
    registry.addOpInterface<VPU::GreaterEqualOp, SwLayerTilingInfoOpModel<VPU::GreaterEqualOp>>();
    registry.addOpInterface<VPU::LogicalOrOp, SwLayerTilingInfoOpModel<VPU::LogicalOrOp>>();
    registry.addOpInterface<VPU::LogicalXorOp, SwLayerTilingInfoOpModel<VPU::LogicalXorOp>>();
    registry.addOpInterface<VPU::LogicalNotOp, SwLayerTilingInfoOpModel<VPU::LogicalNotOp>>();
    registry.addOpInterface<VPU::AndOp, SwLayerTilingInfoOpModel<VPU::AndOp>>();
    registry.addOpInterface<VPU::RoundOp, SwLayerTilingInfoOpModel<VPU::RoundOp>>();
    registry.addOpInterface<VPU::SelectOp, SwLayerTilingInfoOpModel<VPU::SelectOp>>();
    registry.addOpInterface<VPU::ErfOp, SwLayerTilingInfoOpModel<VPU::ErfOp>>();
    registry.addOpInterface<VPU::SinOp, SwLayerTilingInfoOpModel<VPU::SinOp>>();
    registry.addOpInterface<VPU::SinhOp, SwLayerTilingInfoOpModel<VPU::SinhOp>>();
    registry.addOpInterface<VPU::SignOp, SwLayerTilingInfoOpModel<VPU::SignOp>>();
    registry.addOpInterface<VPU::CoshOp, SwLayerTilingInfoOpModel<VPU::CoshOp>>();
    registry.addOpInterface<VPU::TanOp, SwLayerTilingInfoOpModel<VPU::TanOp>>();
    registry.addOpInterface<VPU::SwishOp, SwLayerTilingInfoOpModel<VPU::SwishOp>>();
    registry.addOpInterface<VPU::NegativeOp, SwLayerTilingInfoOpModel<VPU::NegativeOp>>();
    registry.addOpInterface<VPU::CeilingOp, SwLayerTilingInfoOpModel<VPU::CeilingOp>>();
    registry.addOpInterface<VPU::AbsOp, SwLayerTilingInfoOpModel<VPU::AbsOp>>();
    registry.addOpInterface<VPU::SoftMaxOp, SwLayerTilingInfoOpModel<VPU::SoftMaxOp>>();
    registry.addOpInterface<VPU::TopKOp, SwLayerTilingInfoOpModel<VPU::TopKOp>>();
    registry.addOpInterface<VPU::StridedSliceOp, SwLayerTilingInfoOpModel<VPU::StridedSliceOp>>();
    registry.addOpInterface<VPU::SpaceToDepthOp, SwLayerTilingInfoOpModel<VPU::SpaceToDepthOp>>();
    registry.addOpInterface<VPU::DepthToSpaceOp, SwLayerTilingInfoOpModel<VPU::DepthToSpaceOp>>();
    registry.addOpInterface<VPU::YuvToRgbOp, SwLayerTilingInfoOpModel<VPU::YuvToRgbOp>>();
    registry.addOpInterface<VPU::SquaredDifferenceOp, SwLayerTilingInfoOpModel<VPU::SquaredDifferenceOp>>();
    registry.addOpInterface<VPU::GeluOp, SwLayerTilingInfoOpModel<VPU::GeluOp>>();
    registry.addOpInterface<VPU::GridSampleOp, SwLayerTilingInfoOpModel<VPU::GridSampleOp>>();
    registry.addOpInterface<VPU::SoftPlusOp, SwLayerTilingInfoOpModel<VPU::SoftPlusOp>>();

    registry.addOpInterface<VPU::SigmoidOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::HardSigmoidOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::GridSampleOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::SoftMaxOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::HSwishOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::MVNOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::InterpolateOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::ScatterNDUpdateOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::StridedSliceOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::EluOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::SeluOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::ClampOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::FullyConnectedOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::SqrtOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::CeilingOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::NormalizeIEOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::CumSumOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::DivideOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::MultiplyOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::AddOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::SubtractOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::PowerOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::MinimumOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::MaximumOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::ExpOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::RegionYoloOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::GatherOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::GatherElementsOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::GatherNDOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::TanOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::TanhOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::SinOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::CosOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::SinhOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::CoshOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::AsinOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::AcosOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::AtanOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::AsinhOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::AcoshOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::AtanhOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::TopKOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::LRNOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::MemPermuteOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::ConvertOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::PadOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::DepthToSpaceOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::SpaceToDepthOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::AvgPoolOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::FakeQuantizeOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::QuantizeOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::DequantizeOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::PReluOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::ExtractImagePatchesOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::LeakyReluOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::MishOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::TileOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::ReLUOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::YuvToRgbOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::ScatterUpdateOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::ReverseSequenceOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::FloorModOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::EqualOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::GreaterOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::GreaterEqualOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::LessOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::LessEqualOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::LogicalOrOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::HSigmoidOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::LogicalXorOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::LogicalNotOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::AndOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::NotEqualOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::ReduceL1Op, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::ReduceSumOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::ReduceMeanOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::ReduceLogicalAndOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::ReduceMaxOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::ReduceMinOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::ReduceLogicalOrOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::ReduceL2Op, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::ReduceProdOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::NegativeOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::NonMaxSuppressionOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::ROIPoolingOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::PermuteQuantizeOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::LogOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::FloorOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::RoundOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::SignOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::SwishOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::SelectOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::GRUSequenceOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::LSTMCellOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::LSTMSequenceOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::ErfOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::MaxPoolOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::CTCGreedyDecoderSeqLenOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::AbsOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::SquaredDifferenceOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::CTCGreedyDecoderOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::GeluOp, SoftwareLayerOpModel>();
    registry.addOpInterface<VPU::SoftPlusOp, SoftwareLayerOpModel>();

    // When implementing a new SW core, remove the corresponding operation from setupExtraInterfacesAdditional

    redirectOpInterfacesForIE<LayoutInfoOpModelForHW, LayoutInfoOpModelForSW>(registry);
    redirectOpInterfacesForVPUIP<AsyncLayerOpModelForDMA, AsyncLayerOpModelForSW>(registry);

    registry.addAttrInterface<mlir::BuiltinDialect, VPUIP::MemRefAttr, VPUIP::MemRefAttrLayout>();
}

//
// setupExtraInterfacesAdditional
//

void vpux::VPUIP::VPUIPDialect::setupExtraInterfacesAdditional(mlir::DialectRegistry& registry) {
    registry.addOpInterface<VPU::AdaptiveAvgPoolOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::AdaptiveMaxPoolOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::ConvolutionOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::GroupConvolutionOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::ClampOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::ErfOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::BroadcastOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::BucketizeOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::RoundOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::LogOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::YuvToRgbOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::FloorModOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::GRNOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::LRN_IEOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::TileOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::PerAxisTileOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::NegativeOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::PSROIPoolingOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::ProposalOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::DetectionOutputOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::ScaleShiftOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::RegionYoloOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::ReorgYoloOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::CeilingOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::EqualOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::UpsamplingOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::LessOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::LessEqualOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::NotEqualOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::GreaterOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::GreaterEqualOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::AndOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::LogicalNotOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::LogicalOrOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::LogicalXorOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::SpaceToDepthOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::AbsOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::AtanOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::AsinOp, DummySoftwareLayerOpModel>();
    registry.addOpInterface<VPU::AcosOp, DummySoftwareLayerOpModel>();
}

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPUIP/generated/ops.cpp.inc>
