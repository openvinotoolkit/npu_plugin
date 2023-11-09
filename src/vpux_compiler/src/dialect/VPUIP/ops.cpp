//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/core/attributes/indexed_symbol_attr.hpp"
#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"

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

    //
    // Generic HW constrains per VPUX device:
    //

    // On VPUX30XX there is no HW support for AvgPool
    if (arch == VPU::ArchKind::VPUX30XX && mlir::isa<IE::AvgPoolOp>(mainOp)) {
        return false;
    }

    // On VPUX37XX there is no HW support for eltwise And and Mulitiply
    if ((arch == VPU::ArchKind::VPUX37XX) && mlir::isa<IE::AndOp, IE::MultiplyOp>(mainOp)) {
        return false;
    }

    //
    // Activation specific constrains:
    //

    if (mlir::isa<IE::SigmoidOp, IE::TanhOp>(postOp)) {
        // These ops do not get fused for float cases to avoid dropping accuracy. Because PWL is not accurate for FP16
        if (arch == VPU::ArchKind::VPUX30XX) {
            if (!isQuantized(mainOp, postOp)) {
                return false;
            }
        } else if (arch == VPU::ArchKind::VPUX37XX) {
            // VPUX37XX doesn't support sigmoid and tanh
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

    auto leakyReluOp = mlir::dyn_cast<IE::LeakyReluOp>(postOp);
    // Check below is for VPUX30XX case
    if (leakyReluOp != nullptr && arch == VPU::ArchKind::VPUX30XX) {
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
                    getQuantizedType(outLoConst.getContentAttr(), outHiConst.getContentAttr(), fakeQuantizeOp.levels(),
                                     realElemType, false, fakeQuantizeOp.getLoc(), fakeQuantizeOp.auto_broadcast());
            return outElemType.dyn_cast<mlir::quant::UniformQuantizedType>();
        };

        // On VPUX30XX if mainOp is Add, Multiply or And then the PPE is already busy and cannot do LeakyRelu
        if (mlir::isa<IE::AddOp, IE::AndOp, IE::MultiplyOp>(mainOp)) {
            return false;
        }

        const auto reluSlope = leakyReluOp.negative_slopeAttr().getValueAsDouble();
        if (reluSlope < 0) {
            return false;
        }

        if (!isQuantized(mainOp, postOp)) {
            return true;
        }

        if (!leakyReluOp.output().hasOneUse()) {
            return false;
        }

        const auto fqOp = mlir::dyn_cast<IE::FakeQuantizeOp>(*(leakyReluOp.output().getUsers().begin()));
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
// SEOpInterface
//

template <class MainOpType>
class SEOpModel final : public IE::SEOpInterface::ExternalModel<SEOpModel<MainOpType>, MainOpType> {};

//
// TilingInfoOpModel
//

template <class ConcreteOp>
bool isSupportedIsolatedTilingConvBased(ConcreteOp origOp, const OutputTiling& tiles, Logger log) {
    const auto inputType = origOp.input().getType().template cast<NDTypeInterface>();
    const auto outputType = origOp.output().getType().template cast<NDTypeInterface>();
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
    VPUX_THROW_WHEN(nceOp == nullptr, "Op {0} is not NCE", origOp->getLoc());
    auto filterType = nceOp.getWeightsOperand().getType().template cast<NDTypeInterface>();

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        const auto inputTiles = origOp.backInferTileInfo(outputTile, log).tiles;

        VPUX_THROW_UNLESS(inputTiles.size() > 1, "Missed tile information. Got {0} tiles info, must be at least 2",
                          inputTiles.size());
        const auto& inputTile = inputTiles[0];
        const auto& filterTile = inputTiles[1];

        const auto inputTileType = inputType.extractDenseTile(inputTile.offsets, inputTile.shape);
        const auto filterTileType = filterType.extractDenseTile(filterTile.offsets, filterTile.shape);
        const auto outputTileType = outputType.extractDenseTile(outputTile.offsets, outputTile.shape);

        if (origOp->hasAttr(vpux::VPU::multiClusterStrategy)) {
            auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
            VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                            origOp->getLoc());

            auto numClusters = vpux::VPU::getOptimalNumClusters(clusteredOp, outputTileType.getShape()[Dims4D::Act::C],
                                                                clusteredOp.getMultiClusterStrategy().value());
            return origOp.fitIntoCMX(VPU::getDistributedActivationTypeFromOp(clusteredOp, inputTileType, numClusters),
                                     VPU::getDistributedFilterTypeFromOp(nceOp, filterTileType, numClusters),
                                     VPU::getDistributedOutputTypeFromOp(clusteredOp, outputTileType, numClusters));
        }
        return origOp.fitIntoCMX(inputTileType, filterTileType, outputTileType);
    });
}

bool isSupportedIsolatedTiling(VPU::NCEConvolutionOp origOp, const OutputTiling& tiles, Logger log) {
    return isSupportedIsolatedTilingConvBased(origOp, tiles, log);
}

bool isSupportedIsolatedTiling(VPU::NCEInterpolateOp origOp, const OutputTiling& tiles, Logger log) {
    return isSupportedIsolatedTilingConvBased(origOp, tiles, log);
}

bool isSupportedIsolatedTiling(VPU::NCECompressConvolutionOp origOp, const OutputTiling& tiles, Logger log) {
    return isSupportedIsolatedTilingConvBased(origOp, tiles, log);
}

bool isSupportedIsolatedTiling(VPU::NCEDepthConvolutionOp origOp, const OutputTiling& tiles, Logger log) {
    return isSupportedIsolatedTilingConvBased(origOp, tiles, log);
}

bool isSupportedIsolatedTiling(VPU::GroupConvolutionOp origOp, const OutputTiling& tiles, Logger /*log*/) {
    const auto inputType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto filterType = origOp.filter().getType().cast<vpux::NDTypeInterface>();
    const auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    const auto origGroups = origOp.groups().value_or(1);

    const auto origInputShape = getShape(origOp.input());
    const auto origFilterShape = getShape(origOp.filter());
    const auto origBiasShape = origOp.bias() != nullptr ? getShape(origOp.bias()) : ShapeRef();
    const auto origPadding = PadInfo(origOp.pads_begin(), origOp.pads_end());
    const auto numOutChannelsPerGroup = origFilterShape[Dims4D::Filter::OC] / origGroups;

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        // Tiling over output channels should not slice in the middle of a group. Each of the resulting GroupConvs after
        // tiling must have the same number of output channels per group.
        // E.g. GroupConv groups = 5; in channels = 10; out channels = 15; filter = (groups * 3 out ch) x 2 in ch
        //      w/ tiling = [1, 3, 1, 1]
        //      Tile 0: GroupConv groups = 2; in channels = 4; out channels = 5; filter = 5 out ch x 2 in ch
        //              => invalid since group 0 has 3 output channels, while group 1 has 2 output channels

        // An exception for that is when the resulting GroupConv has only one group. Then we can allow it to avoid
        // having to tile on another dim as well.
        // E.g. GroupConv groups = 2; in channels = 10; out channels = 4; filter = (groups * 2 out ch) x 5 in ch
        //      w/ tiling = [1, 4, 1, 1]
        //      Tile 0: GroupConv groups = 1; in channels = 5 (orig channels 0 -> 4); out channels = 1 (orig channel 0);
        //              filter = (groups * 1 out ch) x 5 in ch
        //      Tile 1: GroupConv groups = 1; in channels = 5 (orig channels 0 -> 4); out channels = 1 (orig channel 1);
        //              filter = (groups * 1 out ch) x 5 in ch
        //      Tile 2: GroupConv groups = 1; in channels = 5 (orig channels 5 -> 9); out channels = 1 (orig channel 2);
        //              filter = (groups * 1 out ch) x 5 in ch
        //      Tile 3: GroupConv groups = 1; in channels = 5 (orig channels 5 -> 9); out channels = 1 (orig channel 3);
        //              filter = (groups * 1 out ch) x 5 in ch

        if (outputTile.axis[Dims4D::Act::C] != 1 && outputTile.shape[Dims4D::Act::C] > numOutChannelsPerGroup) {
            if (outputTile.shape[Dims4D::Act::C] % numOutChannelsPerGroup != 0 ||
                outputTile.offsets[Dims4D::Act::C] % numOutChannelsPerGroup != 0) {
                return false;
            }
        }

        const auto inputTiling = backInferGroupConvTile(outputTile, origInputShape, origFilterShape, origBiasShape,
                                                        origOp.strides(), origPadding, origGroups);

        const auto& tileConf = inputTiling.tiles;
        VPUX_THROW_UNLESS(tileConf.size() > 1, "Missed tile information. Got {0} tiles info, must be at least 2",
                          tileConf.size());
        const auto& inputTile = tileConf[0];
        const auto& filterTile = tileConf[1];

        const auto inputTileType = inputType.extractDenseTile(inputTile.offsets, inputTile.shape);
        const auto filterTileType = filterType.extractDenseTile(filterTile.offsets, filterTile.shape);
        const auto outputTileType = outputType.extractDenseTile(outputTile.offsets, outputTile.shape);

        return origOp.fitIntoCMX(inputTileType, filterTileType, outputTileType);
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
                                                          clusteredOp.getMultiClusterStrategy().value());
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
                                                          clusteredOp.getMultiClusterStrategy().value());
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
            isInplace = nceEltwiseOp.is_inplace().value_or(false);
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

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        vpux::OutputTiling inputTiles{outputTile};
        if (auto tilingBuilderInterface = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(origOp.getOperation())) {
            inputTiles = tilingBuilderInterface.backInferTileInfo(outputTile, log).tiles;
        }

        VPUX_THROW_UNLESS(inputTiles.size() == origOp->getOperands().size(),
                          "Unexpect inputTile size '{0}' and Op operands size '{1}'", inputTiles.size(),
                          origOp->getOperands().size());

        mlir::SmallVector<vpux::NDTypeInterface> inputTileTypes;
        for (auto input : origOp->getOperands() | indexed) {
            const auto inputType = input.value().getType().cast<vpux::NDTypeInterface>();
            inputTileTypes.push_back(
                    inputType.extractDenseTile(inputTiles[input.index()].offsets, inputTiles[input.index()].shape));
        }

        auto valueTypes = inputTileTypes;
        mlir::SmallVector<vpux::NDTypeInterface> outputTileTypes;
        for (const auto& output : origOp->getResults()) {
            const auto outputType = output.getType().cast<vpux::NDTypeInterface>();
            const auto outputTileType = outputType.extractDenseTile(outputTile.offsets, outputTile.shape);
            outputTileTypes.push_back(outputTileType);
            valueTypes.push_back(outputTileType);
        }

        if (origOp->hasAttr(VPU::multiClusterStrategy)) {
            auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
            VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                            origOp->getLoc());
            auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTileTypes[0].getShape()[Dims4D::Act::C],
                                                          clusteredOp.getMultiClusterStrategy().value());

            if (!llvm::all_of(outputTileTypes, [&](const vpux::NDTypeInterface& outputTileType) {
                    auto numClustersOfPerOutput =
                            VPU::getOptimalNumClusters(clusteredOp, outputTileType.getShape()[Dims4D::Act::C],
                                                       clusteredOp.getMultiClusterStrategy().value());
                    return numClustersOfPerOutput == numClusters;
                })) {
                return false;
            }

            SmallVector<vpux::NDTypeInterface> distributedTensorTypes;
            for (auto inputTileType : inputTileTypes) {
                auto inDistributedType =
                        VPU::getDistributedActivationTypeFromOp(clusteredOp, inputTileType, numClusters);
                distributedTensorTypes.push_back(inDistributedType.cast<vpux::NDTypeInterface>());
            }

            for (const auto& outputTileType : outputTileTypes) {
                auto outDistributedType = VPU::getDistributedOutputTypeFromOp(clusteredOp, outputTileType, numClusters);
                distributedTensorTypes.push_back(outDistributedType.cast<vpux::NDTypeInterface>());
            }

            return origOp.fitIntoCMX(distributedTensorTypes, Byte(0));
        }

        return origOp.fitIntoCMX(valueTypes, Byte(0));
    });
}

bool isSupportedIsolatedTilingGRUSequence(VPU::GRUSequenceOp op, const OutputTiling& tiles, Logger log) {
    const auto origOp = op.getOperation();

    const auto operands = origOp->getOperands();
    const auto results = origOp->getResults();

    auto tilingOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(origOp);
    VPUX_THROW_UNLESS(tilingOp != nullptr, "Not a tileable operation {0}", origOp->getName());
    const auto cmxAvailableBytes = vpux::VPU::getTotalCMXSize(origOp).to<Byte>().count();

    auto outputYType = results[0].getType().cast<vpux::NDTypeInterface>();
    auto outputYByteSize = outputYType.getElemTypeSize().to<Byte>().count();

    auto seqLength = op.seq_lengthAttr().dyn_cast_or_null<mlir::IntegerAttr>().getValue().getSExtValue();

    return llvm::all_of(tiles, [&](const TileInfo& outputYTile) {
        auto inputTiles = tilingOp.backInferTileInfo(outputYTile, log);
        if (inputTiles.tiles.size() < 1) {
            log.trace("No input tiles for {0}", origOp->getLoc());
            return false;
        }

        const auto outputTileSizeBytes = outputYTile.shape.totalSize() * outputYByteSize +
                                         outputYTile.shape.totalSize() / seqLength * outputYByteSize;
        log.trace("outputTileSizeBytes: {0}", outputTileSizeBytes);
        const auto& inTiles = inputTiles.tiles;
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
                    origOp->getLoc(), inTiles[0].shape, outputYTile.shape, requiredCMX, cmxAvailableBytes);
            return false;
        }
        log.trace("Op {0} out tiling probe valid: {1} - input tile on 0 pos: {2}", origOp->getLoc(), outputYTile,
                  inTiles[0]);
        return true;
    });
}

bool isSupportedIsolatedTilingGeneric(mlir::Operation* origOp, const OutputTiling& firstOutputTiles, Logger log) {
    auto tilingOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(origOp);
    VPUX_THROW_UNLESS(tilingOp != nullptr, "Not a tileable operation {0}", origOp->getName());

    const auto cmxAvailableBytes = vpux::VPU::getTotalCMXSize(origOp).to<Byte>().count();

    const auto operands = origOp->getOperands();
    const auto results = origOp->getResults();

    const auto inputOutputTilesFitCMX = [&](const TileInfo& firstOutputTile) {
        const auto computeRequiredMemory = [&](const auto& values, const SmallVector<TileInfo>& tilingInfo) {
            auto requiredBytes = 0;
            for (const auto& p : zip(values, tilingInfo)) {
                const auto inputType = std::get<0>(p).getType().template cast<vpux::NDTypeInterface>();
                const auto elementByteSize = inputType.getElemTypeSize().template to<Byte>().count();
                const auto tileShape = std::get<1>(p).shape;
                log.trace("tileShape {0}", tileShape);
                requiredBytes += tileShape.totalSize() * elementByteSize;
            }
            return requiredBytes;
        };

        const auto inputTilingInfo = tilingOp.backInferTileInfo(firstOutputTile, log);
        const auto outputTiles = tilingOp.getOutputTiling(firstOutputTile, log);

        const auto inputMemorySize = computeRequiredMemory(operands, inputTilingInfo.tiles);
        const auto outputMemorySize = computeRequiredMemory(results, outputTiles);

        const auto requiredCMX = inputMemorySize + outputMemorySize;

        if (requiredCMX > cmxAvailableBytes) {
            log.trace("Op '{0}' doesn't fit into CMX: required {1}, available {2}", origOp->getLoc(), requiredCMX,
                      cmxAvailableBytes);
            return false;
        }

        return true;
    };

    return llvm::all_of(firstOutputTiles, inputOutputTilesFitCMX);
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

bool isSupportedIsolatedTiling(VPU::NCEPermuteOp origOp, const OutputTiling& tiles, Logger log) {
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
            auto numClusters = VPU::getOptimalNumClusters(
                    clusteredOp, outputTileType.getShape()[Dims4D::Act::H],
                    clusteredOp->getAttr(VPU::multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue());
            return origOp.fitIntoCMX(VPU::getDistributedActivationTypeFromOp(clusteredOp, inputTileType, numClusters),
                                     VPU::getDistributedOutputTypeFromOp(clusteredOp, outputTileType, numClusters));
        }
        return origOp.fitIntoCMX(inputTileType, outputTileType);
    });
}

bool isSupportedIsolatedTilingSwLayer(mlir::Operation* origOp, const OutputTiling& tiles, Logger log) {
    return llvm::TypeSwitch<mlir::Operation*, bool>(origOp)
            .Case<VPU::GroupConvolutionOp>([&](VPU::GroupConvolutionOp op) {
                return isSupportedIsolatedTiling(op, tiles, log);
            })
            .Case<VPU::AddOp, VPU::MultiplyOp, VPU::SubtractOp, VPU::AndOp>([&](mlir::Operation* op) {
                return isSupportedIsolatedTilingEltwise(op, tiles, log);
            })
            .Case<VPU::SWOpInterface>([&](VPU::SWOpInterface swOp) {
                return isSupportedIsolatedTilingSwInterface(swOp, tiles, log);
            })
            .Case<VPU::GRUSequenceOp>([&](VPU::GRUSequenceOp op) {
                return isSupportedIsolatedTilingGRUSequence(op, tiles, log);
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
    if (mlir::failed(tileResult)) {
        return false;
    }
    auto lastTile = tileResult.value().end() - 1;
    auto firstTile = tileResult.value().begin();
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

template <class ConcreteOp>
bool isSupportedPrefetchTilingConvBased(ConcreteOp origOp, const OutputTiling& tiles, Logger log,
                                        TilingMode tilingMode) {
    auto outputShape = getShape(origOp.output());
    auto tileAxis = tiles.front().axis;
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

bool isSupportedPrefetchTiling(VPU::NCEConvolutionOp origOp, const OutputTiling& tiles, Logger log,
                               TilingMode tilingMode) {
    return isSupportedPrefetchTilingConvBased(origOp, tiles, log, tilingMode);
}

bool isSupportedPrefetchTiling(VPU::NCEInterpolateOp origOp, const OutputTiling& tiles, Logger log,
                               TilingMode tilingMode) {
    return isSupportedPrefetchTilingConvBased(origOp, tiles, log, tilingMode);
}

bool isSupportedPrefetchTiling(VPU::NCECompressConvolutionOp origOp, const OutputTiling& tiles, Logger log,
                               TilingMode tilingMode) {
    return isSupportedPrefetchTilingConvBased(origOp, tiles, log, tilingMode);
}

bool isSupportedPrefetchTiling(VPU::NCEDepthConvolutionOp origOp, const OutputTiling& tiles, Logger log,
                               TilingMode tilingMode) {
    return isSupportedPrefetchTilingConvBased(origOp, tiles, log, tilingMode);
}

bool isSupportedPrefetchTiling(VPU::NCEMaxPoolOp origOp, const OutputTiling& tiles, Logger log, TilingMode tilingMode) {
    auto tileAxis = tiles.front().axis;
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
    auto tileAxis = tiles.front().axis;
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

bool isSupportedPrefetchTiling(VPU::NCEPermuteOp origOp, const OutputTiling& /*tiles*/, Logger log,
                               TilingMode /*tilingMode*/) {
    // NCE.NCEPermuteOp will be lowered to eltwise add, same rules are applied.
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
// redirectOpInterfacesForVPUIP
//

template <class OpModelForDMA, class OpModelForSW>
void redirectOpInterfacesForVPUIP(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPUIP::VPUIPDialect*) {
        VPUIP::CopyOp::attachInterface<OpModelForDMA>(*ctx);
        VPUIP::TimestampOp::attachInterface<OpModelForDMA>(*ctx);
        VPUIP::DepthToSpaceDMAOp::attachInterface<OpModelForDMA>(*ctx);
        VPUIP::PermuteDMAOp::attachInterface<OpModelForDMA>(*ctx);
        VPUIP::ExpandDMAOp::attachInterface<OpModelForDMA>(*ctx);
        VPUIP::ExpandOp::attachInterface<OpModelForDMA>(*ctx);

        VPUIP::ConvertUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::SoftMaxUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::PoolingUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::AdaptiveAvgPoolUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::AdaptiveMaxPoolUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::ReLUUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::SigmoidUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::SignUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::ClampUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::EluUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::HSwishUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::MishUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::ErfUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::BroadcastUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::FloorUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::RoundUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::TanUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::TanhUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::SinUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::CosUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::SqrtUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::SinhUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::SeluUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::CoshUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::AsinhUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::AcoshUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::AtanhUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::LogUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::GeluUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::BucketizeUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::QuantCastUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::FakeQuantizeUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::GatherUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::GatherNDUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::ScatterUpdateUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::YuvToRgbUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::GatherElementsUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::ScatterNDUpdateUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::PReluUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::LeakyReluUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::SwishUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::GRNUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::NormUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::ReduceUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::PerAxisTileUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::NegativeUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::ROIPoolingUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::PSROIPoolingUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::ROIAlignUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::ProposalUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::FullyConnectedUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::DetectionOutputUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::ScaleShiftUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::CTCGreedyDecoderUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::CTCGreedyDecoderSeqLenUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::PadUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::ExpUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::InterpolateUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::LSTMCellUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::StridedSliceUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::RegionYoloUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::ReorgYoloUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::MVNUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::LSTMSequenceUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::PermuteUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::CeilingUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::NormalizeIEUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::CumSumUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::SelectUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::DepthToSpaceUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::ReverseSequenceUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::UpsamplingUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::SoftPlusUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::TopKUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::LogicalNotUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::SpaceToDepthUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::ExtractImagePatchesUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::AbsUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::HSigmoidUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::RollUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::AtanUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::AsinUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::AcosUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::HardSigmoidUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::EmbeddingBagOffsetsSumUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::EmbeddingSegmentsSumUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::DeformablePSROIPoolingUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::NonMaxSuppressionUPAOp::attachInterface<OpModelForSW>(*ctx);
        VPUIP::LSTMCellUPAOp::attachInterface<OpModelForSW>(*ctx);
    });
}

}  // namespace

//
// setupExtraInterfaces
//

void vpux::VPUIP::VPUIPDialect::setupExtraInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, IE::IEDialect*) {
        IE::ConvolutionOp::attachInterface<LayerWithPostOpModel<IE::ConvolutionOp>>(*ctx);
        IE::DeconvolutionOp::attachInterface<LayerWithPostOpModel<IE::DeconvolutionOp>>(*ctx);
        IE::GroupConvolutionOp::attachInterface<LayerWithPostOpModel<IE::GroupConvolutionOp>>(*ctx);
        IE::MaxPoolOp::attachInterface<LayerWithPostOpModel<IE::MaxPoolOp>>(*ctx);
        IE::AvgPoolOp::attachInterface<LayerWithPostOpModel<IE::AvgPoolOp>>(*ctx);
        IE::AddOp::attachInterface<LayerWithPostOpModel<IE::AddOp>>(*ctx);
        IE::MultiplyOp::attachInterface<LayerWithPostOpModel<IE::MultiplyOp>>(*ctx);
        IE::SubtractOp::attachInterface<LayerWithPostOpModel<IE::SubtractOp>>(*ctx);
        IE::AndOp::attachInterface<LayerWithPostOpModel<IE::AndOp>>(*ctx);

        IE::InterpolateOp::attachInterface<SEOpModel<IE::InterpolateOp>>(*ctx);
    });

    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        VPU::NCEConvolutionOp::attachInterface<NCETilingInfoOpModel<VPU::NCEConvolutionOp>>(*ctx);
        VPU::NCECompressConvolutionOp::attachInterface<NCETilingInfoOpModel<VPU::NCECompressConvolutionOp>>(*ctx);
        VPU::NCEDepthConvolutionOp::attachInterface<NCETilingInfoOpModel<VPU::NCEDepthConvolutionOp>>(*ctx);
        VPU::NCEMaxPoolOp::attachInterface<NCETilingInfoOpModel<VPU::NCEMaxPoolOp>>(*ctx);
        VPU::NCEAveragePoolOp::attachInterface<NCETilingInfoOpModel<VPU::NCEAveragePoolOp>>(*ctx);
        VPU::NCEEltwiseOp::attachInterface<NCEEltwiseTilingInfoOpModel<VPU::NCEEltwiseOp>>(*ctx);
        VPU::NCEPermuteQuantizeOp::attachInterface<NCETilingInfoOpModel<VPU::NCEPermuteQuantizeOp>>(*ctx);
        VPU::NCEInterpolateOp::attachInterface<NCETilingInfoOpModel<VPU::NCEInterpolateOp>>(*ctx);
        VPU::NCEPermuteOp::attachInterface<NCETilingInfoOpModel<VPU::NCEPermuteOp>>(*ctx);

        VPU::ConvolutionOp::attachInterface<SwLayerTilingInfoOpModel<VPU::ConvolutionOp>>(*ctx);
        VPU::GroupConvolutionOp::attachInterface<SwLayerTilingInfoOpModel<VPU::GroupConvolutionOp>>(*ctx);
        VPU::MaxPoolOp::attachInterface<SwLayerTilingInfoOpModel<VPU::MaxPoolOp>>(*ctx);
        VPU::AddOp::attachInterface<SwLayerTilingInfoOpModel<VPU::AddOp>>(*ctx);
        VPU::MultiplyOp::attachInterface<SwLayerTilingInfoOpModel<VPU::MultiplyOp>>(*ctx);
        VPU::SubtractOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SubtractOp>>(*ctx);
        VPU::AndOp::attachInterface<SwLayerTilingInfoOpModel<VPU::AndOp>>(*ctx);
        VPU::InterpolateOp::attachInterface<SwLayerTilingInfoOpModel<VPU::InterpolateOp>>(*ctx);
        VPU::FakeQuantizeOp::attachInterface<SwLayerTilingInfoOpModel<VPU::FakeQuantizeOp>>(*ctx);
        VPU::QuantizeOp::attachInterface<SwLayerTilingInfoOpModel<VPU::QuantizeOp>>(*ctx);
        VPU::DequantizeOp::attachInterface<SwLayerTilingInfoOpModel<VPU::DequantizeOp>>(*ctx);
        VPU::GatherOp::attachInterface<SwLayerTilingInfoOpModel<VPU::GatherOp>>(*ctx);
        VPU::GatherNDOp::attachInterface<SwLayerTilingInfoOpModel<VPU::GatherNDOp>>(*ctx);
        VPU::ConvertOp::attachInterface<SwLayerTilingInfoOpModel<VPU::ConvertOp>>(*ctx);
        VPU::SigmoidOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SigmoidOp>>(*ctx);
        VPU::HSwishOp::attachInterface<SwLayerTilingInfoOpModel<VPU::HSwishOp>>(*ctx);
        VPU::HSigmoidOp::attachInterface<SwLayerTilingInfoOpModel<VPU::HSigmoidOp>>(*ctx);
        VPU::LeakyReluOp::attachInterface<SwLayerTilingInfoOpModel<VPU::LeakyReluOp>>(*ctx);
        VPU::PReluOp::attachInterface<SwLayerTilingInfoOpModel<VPU::PReluOp>>(*ctx);
        VPU::EluOp::attachInterface<SwLayerTilingInfoOpModel<VPU::EluOp>>(*ctx);
        VPU::ClampOp::attachInterface<SwLayerTilingInfoOpModel<VPU::ClampOp>>(*ctx);
        VPU::ReLUOp::attachInterface<SwLayerTilingInfoOpModel<VPU::ReLUOp>>(*ctx);
        VPU::SqrtOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SqrtOp>>(*ctx);
        VPU::ExpOp::attachInterface<SwLayerTilingInfoOpModel<VPU::ExpOp>>(*ctx);
        VPU::TanhOp::attachInterface<SwLayerTilingInfoOpModel<VPU::TanhOp>>(*ctx);
        VPU::DivideOp::attachInterface<SwLayerTilingInfoOpModel<VPU::DivideOp>>(*ctx);
        VPU::FloorOp::attachInterface<SwLayerTilingInfoOpModel<VPU::FloorOp>>(*ctx);
        VPU::MemPermuteOp::attachInterface<SwLayerTilingInfoOpModel<VPU::MemPermuteOp>>(*ctx);
        VPU::AvgPoolOp::attachInterface<SwLayerTilingInfoOpModel<VPU::AvgPoolOp>>(*ctx);
        VPU::PermuteQuantizeOp::attachInterface<SwLayerTilingInfoOpModel<VPU::PermuteQuantizeOp>>(*ctx);
        VPU::LogOp::attachInterface<SwLayerTilingInfoOpModel<VPU::LogOp>>(*ctx);
        VPU::PowerOp::attachInterface<SwLayerTilingInfoOpModel<VPU::PowerOp>>(*ctx);
        VPU::FloorModOp::attachInterface<SwLayerTilingInfoOpModel<VPU::FloorModOp>>(*ctx);
        VPU::EqualOp::attachInterface<SwLayerTilingInfoOpModel<VPU::EqualOp>>(*ctx);
        VPU::LessOp::attachInterface<SwLayerTilingInfoOpModel<VPU::LessOp>>(*ctx);
        VPU::LessEqualOp::attachInterface<SwLayerTilingInfoOpModel<VPU::LessEqualOp>>(*ctx);
        VPU::NotEqualOp::attachInterface<SwLayerTilingInfoOpModel<VPU::NotEqualOp>>(*ctx);
        VPU::GreaterOp::attachInterface<SwLayerTilingInfoOpModel<VPU::GreaterOp>>(*ctx);
        VPU::GreaterEqualOp::attachInterface<SwLayerTilingInfoOpModel<VPU::GreaterEqualOp>>(*ctx);
        VPU::LogicalOrOp::attachInterface<SwLayerTilingInfoOpModel<VPU::LogicalOrOp>>(*ctx);
        VPU::LogicalXorOp::attachInterface<SwLayerTilingInfoOpModel<VPU::LogicalXorOp>>(*ctx);
        VPU::LogicalNotOp::attachInterface<SwLayerTilingInfoOpModel<VPU::LogicalNotOp>>(*ctx);
        VPU::AndOp::attachInterface<SwLayerTilingInfoOpModel<VPU::AndOp>>(*ctx);
        VPU::RoundOp::attachInterface<SwLayerTilingInfoOpModel<VPU::RoundOp>>(*ctx);
        VPU::SelectOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SelectOp>>(*ctx);
        VPU::ErfOp::attachInterface<SwLayerTilingInfoOpModel<VPU::ErfOp>>(*ctx);
        VPU::DetectionOutputDecodeBoxesOp::attachInterface<SwLayerTilingInfoOpModel<VPU::DetectionOutputDecodeBoxesOp>>(
                *ctx);
        VPU::DetectionOutputSelectBoxesOp::attachInterface<SwLayerTilingInfoOpModel<VPU::DetectionOutputSelectBoxesOp>>(
                *ctx);
        VPU::DetectionOutputSortTopKOp::attachInterface<SwLayerTilingInfoOpModel<VPU::DetectionOutputSortTopKOp>>(*ctx);
        VPU::SinOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SinOp>>(*ctx);
        VPU::SinhOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SinhOp>>(*ctx);
        VPU::SignOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SignOp>>(*ctx);
        VPU::CoshOp::attachInterface<SwLayerTilingInfoOpModel<VPU::CoshOp>>(*ctx);
        VPU::TanOp::attachInterface<SwLayerTilingInfoOpModel<VPU::TanOp>>(*ctx);
        VPU::ReduceSumOp::attachInterface<SwLayerTilingInfoOpModel<VPU::ReduceSumOp>>(*ctx);
        VPU::SwishOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SwishOp>>(*ctx);
        VPU::NegativeOp::attachInterface<SwLayerTilingInfoOpModel<VPU::NegativeOp>>(*ctx);
        VPU::CeilingOp::attachInterface<SwLayerTilingInfoOpModel<VPU::CeilingOp>>(*ctx);
        VPU::AbsOp::attachInterface<SwLayerTilingInfoOpModel<VPU::AbsOp>>(*ctx);
        VPU::SoftMaxOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SoftMaxOp>>(*ctx);
        VPU::LogSoftmaxOp::attachInterface<SwLayerTilingInfoOpModel<VPU::LogSoftmaxOp>>(*ctx);
        VPU::TopKOp::attachInterface<SwLayerTilingInfoOpModel<VPU::TopKOp>>(*ctx);
        VPU::StridedSliceOp::attachInterface<SwLayerTilingInfoOpModel<VPU::StridedSliceOp>>(*ctx);
        VPU::SpaceToDepthOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SpaceToDepthOp>>(*ctx);
        VPU::DepthToSpaceOp::attachInterface<SwLayerTilingInfoOpModel<VPU::DepthToSpaceOp>>(*ctx);
        VPU::TileOp::attachInterface<SwLayerTilingInfoOpModel<VPU::TileOp>>(*ctx);
        VPU::NormalizeL2Op::attachInterface<SwLayerTilingInfoOpModel<VPU::NormalizeL2Op>>(*ctx);
        VPU::YuvToRgbOp::attachInterface<SwLayerTilingInfoOpModel<VPU::YuvToRgbOp>>(*ctx);
        VPU::SquaredDifferenceOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SquaredDifferenceOp>>(*ctx);
        VPU::GeluOp::attachInterface<SwLayerTilingInfoOpModel<VPU::GeluOp>>(*ctx);
        VPU::GridSampleOp::attachInterface<SwLayerTilingInfoOpModel<VPU::GridSampleOp>>(*ctx);
        VPU::GRUSequenceOp::attachInterface<SwLayerTilingInfoOpModel<VPU::GRUSequenceOp>>(*ctx);
        VPU::SoftPlusOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SoftPlusOp>>(*ctx);
        VPU::MVNOp::attachInterface<SwLayerTilingInfoOpModel<VPU::MVNOp>>(*ctx);
        VPU::MVN6Op::attachInterface<SwLayerTilingInfoOpModel<VPU::MVN6Op>>(*ctx);
        VPU::DFTOp::attachInterface<SwLayerTilingInfoOpModel<VPU::DFTOp>>(*ctx);
        VPU::RDFTUncutOp::attachInterface<SwLayerTilingInfoOpModel<VPU::RDFTUncutOp>>(*ctx);
        VPU::IDFTOp::attachInterface<SwLayerTilingInfoOpModel<VPU::IDFTOp>>(*ctx);
        VPU::IRDFTLastAxisOp::attachInterface<SwLayerTilingInfoOpModel<VPU::IRDFTLastAxisOp>>(*ctx);
        VPU::HardSigmoidOp::attachInterface<SwLayerTilingInfoOpModel<VPU::HardSigmoidOp>>(*ctx);
        VPU::SigmoidOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::HardSigmoidOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GridSampleOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::SoftMaxOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::LogSoftmaxOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::HSwishOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::MVNOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::MVN1SumOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::MVN1MeanVarOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::MVN1NormalizeOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::MVN6Op::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::InterpolateOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ScatterNDUpdateOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::StridedSliceOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::EluOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::SeluOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ClampOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::FullyConnectedOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::SqrtOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::CeilingOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::NormalizeL2Op::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::CumSumOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::DetectionOutputNormalizeOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::DetectionOutputDecodeBoxesOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::DetectionOutputSortTopKOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::DetectionOutputSelectBoxesOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::DetectionOutputNmsCaffeOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::DetectionOutputCollectResultsOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::DivideOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::MultiplyOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::AddOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::SubtractOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::PowerOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::MinimumOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::MaximumOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ExpOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::RegionYoloOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GatherOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GatherSliceOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ExtractValueOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GatherElementsOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GatherNDOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GatherTreeOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::TanOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::TanhOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::SinOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::CosOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::SinhOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::EmbeddingSegmentsSumOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::CoshOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::AsinOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::AcosOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::AtanOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::AsinhOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::AcoshOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::AtanhOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::TopKOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::LRNOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::MemPermuteOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ConvertOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::PadOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::DepthToSpaceOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::SpaceToDepthOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::AvgPoolOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::AdaptiveAvgPoolOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::FakeQuantizeOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::QuantizeOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::DequantizeOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::DynamicQuantizeOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::PReluOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ExtractImagePatchesOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::LeakyReluOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::MishOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::TileOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ReLUOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::YuvToRgbOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::RandomUniformOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::OneHotOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ReorgYoloOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ProposalOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ScatterUpdateOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ScatterElementsUpdateOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ReverseSequenceOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::FloorModOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::EqualOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GreaterOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GreaterEqualOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::LessOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::LessEqualOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::LogicalOrOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::HSigmoidOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::LogicalXorOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::LogicalNotOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::AndOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::NotEqualOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ReduceL1Op::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ReduceSumOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ReduceMeanOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ReduceLogicalAndOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ReduceMaxOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ReduceMinOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ReduceLogicalOrOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ReduceL2Op::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ReduceProdOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::NegativeOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::NonMaxSuppressionOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ROIPoolingOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::PSROIPoolingOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::PermuteQuantizeOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::LogOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::FloorOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::RoundOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::SignOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::SwishOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::SelectOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::EmbeddingBagOffsetsSumOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GRUSequenceOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::EmbeddingBagPackedSumOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GRUSequenceFirstPartOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GRUSequenceLastPartOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::LSTMCellOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::LSTMGatesOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::LSTMSequenceOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ErfOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::MaxPoolOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::RollOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::CTCGreedyDecoderSeqLenOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::AbsOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::SquaredDifferenceOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::CTCGreedyDecoderOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GeluOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::SoftPlusOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ConvolutionOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GroupConvolutionOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::DFTOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::RDFTOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::IDFTOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::IRDFTOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::RDFTUncutOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::IRDFTLastAxisOp::attachInterface<SoftwareLayerOpModel>(*ctx);
    });

    // When implementing a new SW core, remove the corresponding operation from setupExtraInterfacesAdditional

    redirectOpInterfacesForVPUIP<AsyncLayerOpModelForDMA, AsyncLayerOpModelForSW>(registry);

    registry.addExtension(+[](mlir::MLIRContext* ctx, mlir::BuiltinDialect*) {
        VPUIP::MemRefAttr::attachInterface<VPUIP::MemRefAttrLayout>(*ctx);
    });
}

//
// setupExtraInterfacesAdditional
//

void vpux::VPUIP::VPUIPDialect::setupExtraInterfacesAdditional(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        VPU::AdaptiveMaxPoolOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::ClampOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::ErfOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::BroadcastOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::BucketizeOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::LogOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::YuvToRgbOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::GRNOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::LRN_IEOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::TileOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::PerAxisTileOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::NegativeOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::DetectionOutputOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::ScaleShiftOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::CeilingOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::UpsamplingOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::SpaceToDepthOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
    });
}

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPUIP/ops.cpp.inc>
