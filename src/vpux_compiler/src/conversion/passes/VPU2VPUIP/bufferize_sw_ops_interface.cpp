//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferizable_ops_interface.hpp"
#include "vpux/compiler/conversion/rewriters/VPU2VPUIP/sw_rewriter.hpp"

#include "vpux/compiler/dialect/VPUIP/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/sw_utils.hpp"
#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/utils/core/logger.hpp"

using namespace vpux;

namespace {

//
// getOptimalCMXPlacement
//

std::pair<mlir::DenseSet<size_t>, mlir::DenseSet<size_t>> getOptimalCMXPlacement(ArrayRef<mlir::Value> inputs,
                                                                                 ArrayRef<mlir::Value> outputs,
                                                                                 Byte reservedMem,
                                                                                 mlir::ModuleOp module,
                                                                                 const Logger& log) {
    VPUX_THROW_WHEN(inputs.size() == 0 && outputs.size() == 0, "Received empty input and output arrays!");
    auto nestedLog2 = log.nest(2);
    auto nestedLog3 = log.nest(3);
    // Create an array of all the inputs and outputs and index them
    SmallVector<mlir::Value> mergedVals;
    mergedVals.reserve(inputs.size() + outputs.size());
    mergedVals.insert(mergedVals.end(), inputs.begin(), inputs.end());
    mergedVals.insert(mergedVals.end(), outputs.begin(), outputs.end());
    SmallVector<size_t> idxVec(inputs.size() + outputs.size(), 0);
    std::iota(idxVec.begin(), idxVec.end(), 0);

    mlir::DenseSet<size_t> inputsForCMX;
    mlir::DenseSet<size_t> outputsForCMX;

    SmallVector<Byte> ioCmxSizes;
    for (const auto& val : mergedVals) {
        ioCmxSizes.push_back(val.getType().cast<vpux::NDTypeInterface>().getTotalAllocSize());
    }

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? VPU::getTotalCMXSize(module).count()
                                                          : VPU::getTotalCMXFragmentationAwareSize(module).count();

    nestedLog2.trace("Total available CMX size: {0} bytes", totalAvailableCMXSize);

    Byte defaultCMXOffsetAlignment = Byte(vpux::DEFAULT_CMX_ALIGNMENT);
    Byte defaultCMXSizeAlignment = Byte(1);

    auto requiredSizeForAllIO = vpux::calculateAlignedBuffersMemoryRequirement(ioCmxSizes, defaultCMXOffsetAlignment,
                                                                               defaultCMXSizeAlignment)
                                        .count();
    // If all inputs and outputs already fit in NNCMX, this is already optimal
    if (requiredSizeForAllIO + reservedMem.count() <= totalAvailableCMXSize) {
        nestedLog2.trace("All the inputs and outputs will fit in CMX. Total size: {0} bytes", requiredSizeForAllIO);
        auto inputsSize = inputs.size();
        for (size_t i = 0; i < inputsSize; ++i) {
            inputsForCMX.insert(i);
        }
        auto outputsSize = outputs.size();
        for (size_t i = 0; i < outputsSize; ++i) {
            outputsForCMX.insert(i);
        }
        return {inputsForCMX, outputsForCMX};
    }

    // Not all inputs and outputs fit in the available NNCMX space. Find the maximal subset that fits,
    // such that we use NNCMX as much as possible.
    SmallVector<std::pair<SmallVector<size_t>, int64_t>> subsets;
    SmallVector<size_t> aux;

    auto genSubsets = [](const auto& idxVec, auto& subsets, auto& aux, size_t currentIdx,
                         auto& genSubsetsFunc) -> void {
        subsets.push_back({aux, 0});
        for (size_t i = currentIdx; i < idxVec.size(); ++i) {
            aux.push_back(idxVec[i]);
            genSubsetsFunc(idxVec, subsets, aux, i + 1, genSubsetsFunc);
            aux.pop_back();
        }
    };

    // Generate all subsets
    genSubsets(idxVec, subsets, aux, 0, genSubsets);

    // For each subset, compute the total necesssary NNCMX size
    for (auto& p : subsets) {
        const auto& idxVec = p.first;
        SmallVector<Byte> bufferSizes;
        bufferSizes.reserve(idxVec.size());
        for (const auto& idx : idxVec) {
            bufferSizes.push_back(mergedVals[idx].getType().cast<vpux::NDTypeInterface>().getTotalAllocSize());
        }
        p.second = vpux::calculateAlignedBuffersMemoryRequirement(bufferSizes, defaultCMXOffsetAlignment,
                                                                  defaultCMXSizeAlignment)
                           .count() +
                   reservedMem.count();
    }

    std::sort(subsets.begin(), subsets.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.second < rhs.second;
    });

    // Find the subset that uses the most NNCMX without going over the limit
    size_t subsetIdx;
    for (subsetIdx = 0; subsetIdx < subsets.size(); ++subsetIdx) {
        if (subsets[subsetIdx].second > totalAvailableCMXSize) {
            --subsetIdx;
            break;
        }
    }

    const auto maxValidCmxUsageSubsetIdx = std::min(subsetIdx, subsets.size() - 1);

    for (const auto& idx : subsets[maxValidCmxUsageSubsetIdx].first) {
        if (idx < inputs.size()) {
            inputsForCMX.insert(idx);
        } else {
            outputsForCMX.insert(idx - inputs.size());
        }
    }

    nestedLog2.trace("Following inputs and outputs will be mapped to CMX:");

    nestedLog2.trace("Inputs:");
    for (const auto& i : inputsForCMX) {
        nestedLog3.trace("'{0}'", i);
    }

    nestedLog2.trace("Outputs:");
    for (const auto& o : outputsForCMX) {
        nestedLog3.trace("'{0}'", o);
    }

    return {inputsForCMX, outputsForCMX};
}

//
// isDMAConvertibleSwOp
//

bool isDMAConvertibleSwOp(VPUIP::SoftwareLayerOpInterface swOp) {
    return mlir::isa<VPU::MemPermuteOp, VPU::SpaceToDepthOp, VPU::DepthToSpaceOp, VPU::PerAxisTileOp>(swOp);
}

//
// createBuiltInFunction
//

mlir::SymbolRefAttr createBuiltInFunction(mlir::ModuleOp module, VPU::LayerOpInterface origOp,
                                          ArrayRef<mlir::Value> operands, ArrayRef<mlir::Value> results,
                                          const VPUIP::KernelInfo& kernelInfo, const Logger& log) {
    OpBuilderLogger builderLog(log);

    SmallString builtInFunctionName{VPUIP::SW_KERNEL_NAME_PREFIX};
    auto nonNamespaceOpName = origOp->getName().getStringRef().slice(origOp->getName().getDialectNamespace().size() + 1,
                                                                     mlir::StringRef::npos);
    builtInFunctionName.append(nonNamespaceOpName);

    const auto convertToUnrankedType = [](mlir::Value operand) -> mlir::Type {
        auto type = operand.getType().dyn_cast_or_null<mlir::MemRefType>();
        VPUX_THROW_UNLESS(type != nullptr, "Only MemRef type is supported");

        return mlir::UnrankedMemRefType::get(type.getElementType(), type.getMemorySpace());
    };

    auto& args = kernelInfo.args;

    SmallVector<mlir::Type> inputTypes;
    std::transform(operands.begin(), operands.end(), std::back_inserter(inputTypes), convertToUnrankedType);
    std::transform(results.begin(), results.end(), std::back_inserter(inputTypes), convertToUnrankedType);
    std::transform(args.begin(), args.end(), std::back_inserter(inputTypes), [&module](mlir::Attribute arg) {
        const auto typedAttr = arg.dyn_cast<mlir::TypedAttr>();
        return typedAttr != nullptr ? typedAttr.getType() : mlir::NoneType::get(module.getContext());
    });

    return VPUIP::createBuiltInFunction(module, builtInFunctionName, inputTypes, kernelInfo.entryName,
                                        kernelInfo.sourceFileName, log);
}

//
// createAlloc
//

mlir::Value createAlloc(mlir::Location loc, mlir::RewriterBase& rewriter, mlir::Value value,
                        const mlir::bufferization::BufferizationOptions& options, vpux::IndexedSymbolAttr memSpace) {
    auto bufferType = vpux::getBufferType(value, options);
    auto resultBufferType = bufferType.changeMemSpace(memSpace);
    auto allocOp = rewriter.create<mlir::memref::AllocOp>(loc, resultBufferType.cast<mlir::MemRefType>());
    return allocOp.getMemref();
}

//
// createAlloc using diaclet conversion & createAllocAdaptor & replaceOpAdaptor
// Note: remove after fully implemented one-shot bufferization, E#102424
//

mlir::Value createAlloc(mlir::Location loc, mlir::RewriterBase& rewriter, mlir::Value value,
                        mlir::TypeConverter& typeConverter, vpux::IndexedSymbolAttr memSpace) {
    auto bufferType = typeConverter.convertType(value.getType()).cast<vpux::NDTypeInterface>();
    auto resultBufferType = bufferType.changeMemSpace(memSpace);
    auto allocOp = rewriter.create<mlir::memref::AllocOp>(loc, resultBufferType.cast<mlir::MemRefType>());
    return allocOp.getMemref();
}

mlir::Value createAllocAdaptor(mlir::Location loc, mlir::RewriterBase& rewriter, mlir::Value value,
                               const std::optional<mlir::bufferization::BufferizationOptions>& options,
                               std::optional<std::reference_wrapper<mlir::TypeConverter>> typeConverter,
                               vpux::IndexedSymbolAttr memSpace) {
    return options.has_value() ? createAlloc(loc, rewriter, value, options.value(), memSpace)
                               : createAlloc(loc, rewriter, value, typeConverter.value(), memSpace);
}

void replaceOpAdaptor(const std::optional<mlir::bufferization::BufferizationOptions>& options,
                      mlir::RewriterBase& rewriter, mlir::Operation* op, const SmallVector<mlir::Value>& newResults) {
    options.has_value() ? mlir::bufferization::replaceOpWithBufferizedValues(rewriter, op, newResults)
                        : rewriter.replaceOp(op, newResults);
}

}  // namespace

//
// bufferizeSoftwareLayer
//

mlir::LogicalResult vpux::bufferizeSoftwareLayer(
        mlir::RewriterBase& rewriter, mlir::ModuleOp module, mlir::Operation* op, ArrayRef<mlir::Value> newOperands,
        const std::optional<mlir::bufferization::BufferizationOptions>& options,
        std::optional<std::reference_wrapper<mlir::TypeConverter>> typeConverter, vpux::Logger log) {
    VPUX_THROW_WHEN(((options.has_value() && typeConverter.has_value()) ||
                     (!options.has_value() && !typeConverter.has_value())),
                    "Only one of BufferizationOptions and TypeConverter can be set at a time.");
    auto* ctx = op->getContext();
    auto layerOp = mlir::cast<VPU::LayerOpInterface>(op);
    auto swLayerOp = mlir::cast<VPUIP::SoftwareLayerOpInterface>(op);
    const auto memSpaceCMX = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN), 0);

    SmallVector<mlir::Value> opResults(op->getResults().begin(), op->getResults().end());
    auto idxForCMX = getOptimalCMXPlacement(newOperands, opResults, Byte(0), module, log);

    SmallVector<mlir::Value> swKernelOperands;
    for (size_t i = 0; i < newOperands.size(); ++i) {
        if (idxForCMX.first.count(i) == 0) {
            // Operand should remain in DDR according to mapping
            swKernelOperands.push_back(newOperands[i]);
        } else {
            log.trace("Create CMX buffer and copy operation for input: {0}", newOperands[i].getLoc());
            const auto outputBuffer = createAllocAdaptor(newOperands[i].getLoc(), rewriter, newOperands[i], options,
                                                         typeConverter, memSpaceCMX);
            auto copyOp = rewriter.create<VPUIP::CopyOp>(op->getLoc(), newOperands[i], outputBuffer);
            swKernelOperands.push_back(copyOp.getOutput());
        }
    }

    SmallVector<mlir::Value> swKernelResults;
    for (size_t i = 0; i < op->getResults().size(); ++i) {
        if (idxForCMX.second.count(i) == 0) {
            log.trace("Create DDR buffer for output: {0}", op->getResults()[i].getLoc());
            const auto outputBuffer =
                    createAllocAdaptor(op->getLoc(), rewriter, op->getResults()[i], options, typeConverter, nullptr);
            swKernelResults.push_back(outputBuffer);
        } else {
            log.trace("Create CMX buffer for output: {0}", op->getResults()[i].getLoc());
            const auto outputBuffer = createAllocAdaptor(op->getLoc(), rewriter, op->getResults()[i], options,
                                                         typeConverter, memSpaceCMX);
            swKernelResults.push_back(outputBuffer);
        }
    }

    VPUIP::createRuntimeKernelDefinition(module, log.nest());

    // TODO : tile 0
    const int64_t tileIndex = 0;
    auto builtInFunction = createBuiltInFunction(module, layerOp, swKernelOperands, swKernelResults,
                                                 swLayerOp.getKernelInfo(), log.nest());
    auto swKernelOp = rewriter.create<VPUIP::SwKernelOp>(op->getLoc(), swKernelOperands, swKernelResults,
                                                         builtInFunction, getIntAttr(ctx, tileIndex));

    vpux::VPUIP::initSwKernel(swKernelOp, swKernelOperands, swKernelResults, swLayerOp.getKernelInfo().args,
                              log.nest());

    const auto moveSwOpToCMX = [&]() {
        // Go through all inputs and outputs that were mapped to DDR and map them to NNCMX
        if (idxForCMX.first.size() == swKernelOp.getInputs().size() &&
            idxForCMX.second.size() == swKernelOp.getResults().size()) {
            return;
        }

        SmallVector<mlir::Value> cmxOperands;
        cmxOperands.reserve(swKernelOperands.size());
        if (idxForCMX.first.size() != swKernelOp.getInputs().size()) {
            for (const auto& operand : swKernelOperands) {
                if (operand.getType().cast<vpux::NDTypeInterface>().getMemSpace() == memSpaceCMX) {
                    cmxOperands.push_back(operand);
                } else {
                    const auto outputBuffer = createAllocAdaptor(operand.getLoc(), rewriter, operand, options,
                                                                 typeConverter, memSpaceCMX);
                    auto copyOp = rewriter.create<VPUIP::CopyOp>(op->getLoc(), operand, outputBuffer);
                    cmxOperands.push_back(copyOp.getOutput());
                }
            }
        } else {
            cmxOperands.append(swKernelOperands.begin(), swKernelOperands.end());
        }

        SmallVector<mlir::Value> cmxResults;
        cmxResults.reserve(swKernelResults.size());
        if (idxForCMX.second.size() != swKernelOp.getResults().size()) {
            for (const auto& result : swKernelResults) {
                cmxResults.push_back(result);
                if (result.getType().cast<vpux::NDTypeInterface>().getMemSpace() != memSpaceCMX) {
                    cmxResults.back().setType(
                            result.getType().cast<vpux::NDTypeInterface>().changeMemSpace(memSpaceCMX));
                }
            }
        } else {
            cmxResults.append(swKernelResults.begin(), swKernelResults.end());
        }

        auto parentModule = swKernelOp->getParentOfType<mlir::ModuleOp>();
        VPUX_THROW_UNLESS(parentModule, "Sw Kernel Op {0} has no parent Module Op", swKernelOp);
        auto kernelFunc = parentModule.lookupSymbol<mlir::func::FuncOp>(swKernelOp.getKernelFunctionAttr());
        if (kernelFunc) {
            kernelFunc.erase();
        }

        rewriter.eraseOp(swKernelOp);

        builtInFunction =
                createBuiltInFunction(module, layerOp, cmxOperands, cmxResults, swLayerOp.getKernelInfo(), log.nest());

        swKernelOp = rewriter.create<VPUIP::SwKernelOp>(op->getLoc(), cmxOperands, cmxResults, builtInFunction,
                                                        getIntAttr(ctx, tileIndex));

        vpux::VPUIP::initSwKernel(swKernelOp, cmxOperands, cmxResults, swLayerOp.getKernelInfo().args, log.nest());
    };

    if (isDMAConvertibleSwOp(mlir::dyn_cast<vpux::VPUIP::SoftwareLayerOpInterface>(op)) &&
        vpux::VPUIP::isLegalAndBeneficialConvertToDMA(swKernelOp, log)) {
        log.trace("SW Kernel will be converted to DMA Operation: {0}", swKernelOp);
        moveSwOpToCMX();
    }

    log.trace("Added kernel operation: {0}", swKernelOp);

    SmallVector<mlir::Value> finalResults;
    for (auto&& result : swKernelOp.getResults()) {
        if (result.getType().cast<vpux::NDTypeInterface>().getMemSpace() == memSpaceCMX) {
            // Copy outputs that were mapped to CMX back to DDR
            log.trace("Create DDR buffer for output: {0}", result.getLoc());
            const auto outputBuffer =
                    createAllocAdaptor(op->getLoc(), rewriter, result, options, typeConverter, nullptr);
            auto copyOp = rewriter.create<VPUIP::CopyOp>(op->getLoc(), result, outputBuffer);
            finalResults.push_back(copyOp.getOutput());
        } else {
            finalResults.push_back(result);
        }
    }

    log.trace("Replace origin op {0} with new outputs from SW Kernel {1}", op->getLoc(), finalResults);
    replaceOpAdaptor(options, rewriter, op, finalResults);
    return mlir::success();
}

//
// bufferizeNceClusterTilingSoftwareLayer
//

mlir::LogicalResult vpux::bufferizeNceClusterTilingSoftwareLayer(
        mlir::RewriterBase& rewriter, mlir::ModuleOp module, mlir::Operation* op, ArrayRef<mlir::Value> newOperands,
        const std::optional<mlir::bufferization::BufferizationOptions>& options,
        std::optional<std::reference_wrapper<mlir::TypeConverter>> typeConverter, vpux::Logger log) {
    VPUX_THROW_WHEN(((options.has_value() && typeConverter.has_value()) ||
                     (!options.has_value() && !typeConverter.has_value())),
                    "Only one of BufferizationOptions and TypeConverter can be set at a time.");

    auto layerOp = mlir::cast<VPU::LayerOpInterface>(op);
    auto swLayerOp = mlir::cast<VPUIP::SoftwareLayerOpInterface>(op);

    VPUIP::createRuntimeKernelDefinition(module, log.nest());

    auto outputBuffers = allocateBuffersAdaptor(log, op->getLoc(), rewriter, op->getResults(), options, typeConverter,
                                                /*individualBuffers=*/true);
    // actual tile index will be corrected as part of unroll NCEClusterTiling pass, this index will be dropped
    const int64_t tileIndex = 0;
    auto builtInFunction =
            createBuiltInFunction(module, layerOp, newOperands, outputBuffers, swLayerOp.getKernelInfo(), log.nest());
    auto swKernelOp = rewriter.create<VPUIP::SwKernelOp>(op->getLoc(), newOperands, outputBuffers, builtInFunction,
                                                         getIntAttr(op->getContext(), tileIndex));
    vpux::VPUIP::initSwKernel(swKernelOp, newOperands, outputBuffers, swLayerOp.getKernelInfo().args, log.nest());
    replaceOpAdaptor(options, rewriter, op, swKernelOp.getResults());
    return mlir::success();
}

//
// SoftwareLayerOpBufferizeModel
//

template <typename MainOpType>
class SoftwareLayerOpBufferizeModel :
        public BufferizableOpInterfaceExternalModelBase<SoftwareLayerOpBufferizeModel<MainOpType>, MainOpType> {
public:
    mlir::LogicalResult bufferizeImpl(MainOpType op, mlir::RewriterBase& rewriter,
                                      const mlir::bufferization::BufferizationOptions& options,
                                      mlir::ArrayRef<mlir::Value> bufferizedOperands) const;
};

template <typename MainOpType>
mlir::LogicalResult SoftwareLayerOpBufferizeModel<MainOpType>::bufferizeImpl(
        MainOpType op, mlir::RewriterBase& rewriter, const mlir::bufferization::BufferizationOptions& options,
        mlir::ArrayRef<mlir::Value> bufferizedOperands) const {
    auto log = Logger::global().nest("one-shot-bufferize-VPU-to-VPUIP", 0);
    log.trace("Got operation {0} at {1}", op->getName(), op->getLoc());

    constexpr bool opIsSwLayerOperation = MainOpType::template hasTrait<VPU::LayerOpInterface::Trait>() ||
                                          MainOpType::template hasTrait<VPUIP::SoftwareLayerOpInterface::Trait>();
    static_assert(opIsSwLayerOperation, "MainOpType is not a Software layer operation");

    auto module = op->template getParentOfType<mlir::ModuleOp>();
    if (!module) {
        return errorAt(op->getLoc(), "Operation {0} has no parent Module Op", op->getName());
    }

    auto clusterTilingOp = op->template getParentOfType<VPU::NCEClusterTilingOp>();
    if (!clusterTilingOp) {
        return bufferizeSoftwareLayer(rewriter, module, op, bufferizedOperands, options, std::nullopt, log);
    } else {
        return bufferizeNceClusterTilingSoftwareLayer(rewriter, module, op, bufferizedOperands, options, std::nullopt,
                                                      log);
    }
}

//
// registerSoftwareLayerBufferizableOpInterfaces
//

void vpux::registerSoftwareLayerBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*, VPUIP::VPUIPDialect*) {
        VPU::ConvertOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ConvertOp>>(*ctx);
        VPU::SigmoidOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SigmoidOp>>(*ctx);
        VPU::HardSigmoidOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::HardSigmoidOp>>(*ctx);
        VPU::GridSampleOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GridSampleOp>>(*ctx);
        VPU::SoftMaxOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SoftMaxOp>>(*ctx);
        VPU::LogSoftmaxOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::LogSoftmaxOp>>(*ctx);
        VPU::HSwishOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::HSwishOp>>(*ctx);
        VPU::MVNOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::MVNOp>>(*ctx);
        VPU::MVN1SumOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::MVN1SumOp>>(*ctx);
        VPU::MVN1MeanVarOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::MVN1MeanVarOp>>(*ctx);
        VPU::MVN1NormalizeOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::MVN1NormalizeOp>>(*ctx);
        VPU::MVN6Op::attachInterface<SoftwareLayerOpBufferizeModel<VPU::MVN6Op>>(*ctx);
        VPU::InterpolateOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::InterpolateOp>>(*ctx);
        VPU::ScatterNDUpdateOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ScatterNDUpdateOp>>(*ctx);
        VPU::StridedSliceOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::StridedSliceOp>>(*ctx);
        VPU::EluOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::EluOp>>(*ctx);
        VPU::SeluOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SeluOp>>(*ctx);
        VPU::ClampOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ClampOp>>(*ctx);
        VPU::FullyConnectedOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::FullyConnectedOp>>(*ctx);
        VPU::SqrtOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SqrtOp>>(*ctx);
        VPU::CeilingOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::CeilingOp>>(*ctx);
        VPU::NormalizeL2Op::attachInterface<SoftwareLayerOpBufferizeModel<VPU::NormalizeL2Op>>(*ctx);
        VPU::CumSumOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::CumSumOp>>(*ctx);
        VPU::EyeOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::EyeOp>>(*ctx);
        VPU::DetectionOutputNormalizeOp::attachInterface<
                SoftwareLayerOpBufferizeModel<VPU::DetectionOutputNormalizeOp>>(*ctx);
        VPU::DetectionOutputDecodeBoxesOp::attachInterface<
                SoftwareLayerOpBufferizeModel<VPU::DetectionOutputDecodeBoxesOp>>(*ctx);
        VPU::DetectionOutputSortTopKOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::DetectionOutputSortTopKOp>>(
                *ctx);
        VPU::DetectionOutputSelectBoxesOp::attachInterface<
                SoftwareLayerOpBufferizeModel<VPU::DetectionOutputSelectBoxesOp>>(*ctx);
        VPU::DetectionOutputNmsCaffeOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::DetectionOutputNmsCaffeOp>>(
                *ctx);
        VPU::DetectionOutputCollectResultsOp::attachInterface<
                SoftwareLayerOpBufferizeModel<VPU::DetectionOutputCollectResultsOp>>(*ctx);
        VPU::DivideOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::DivideOp>>(*ctx);
        VPU::MultiplyOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::MultiplyOp>>(*ctx);
        VPU::AddOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::AddOp>>(*ctx);
        VPU::SubtractOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SubtractOp>>(*ctx);
        VPU::PowerOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::PowerOp>>(*ctx);
        VPU::MinimumOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::MinimumOp>>(*ctx);
        VPU::MaximumOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::MaximumOp>>(*ctx);
        VPU::ExpOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ExpOp>>(*ctx);
        VPU::RegionYoloOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::RegionYoloOp>>(*ctx);
        VPU::GatherOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GatherOp>>(*ctx);
        VPU::GatherElementsOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GatherElementsOp>>(*ctx);
        VPU::GatherNDOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GatherNDOp>>(*ctx);
        VPU::GatherTreeOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GatherTreeOp>>(*ctx);
        VPU::ConditionalCopyOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ConditionalCopyOp>>(*ctx);
        VPU::TanOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::TanOp>>(*ctx);
        VPU::TanhOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::TanhOp>>(*ctx);
        VPU::SinOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SinOp>>(*ctx);
        VPU::CosOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::CosOp>>(*ctx);
        VPU::SinhOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SinhOp>>(*ctx);
        VPU::EmbeddingSegmentsSumOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::EmbeddingSegmentsSumOp>>(*ctx);
        VPU::CoshOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::CoshOp>>(*ctx);
        VPU::AsinOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::AsinOp>>(*ctx);
        VPU::AcosOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::AcosOp>>(*ctx);
        VPU::AtanOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::AtanOp>>(*ctx);
        VPU::AsinhOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::AsinhOp>>(*ctx);
        VPU::AcoshOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::AcoshOp>>(*ctx);
        VPU::AtanhOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::AtanhOp>>(*ctx);
        VPU::TopKOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::TopKOp>>(*ctx);
        VPU::LRNOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::LRNOp>>(*ctx);
        VPU::MemPermuteOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::MemPermuteOp>>(*ctx);
        VPU::PadOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::PadOp>>(*ctx);
        VPU::DepthToSpaceOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::DepthToSpaceOp>>(*ctx);
        VPU::SpaceToDepthOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SpaceToDepthOp>>(*ctx);
        VPU::AvgPoolOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::AvgPoolOp>>(*ctx);
        VPU::AdaptiveAvgPoolOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::AdaptiveAvgPoolOp>>(*ctx);
        VPU::FakeQuantizeOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::FakeQuantizeOp>>(*ctx);
        VPU::QuantizeOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::QuantizeOp>>(*ctx);
        VPU::DequantizeOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::DequantizeOp>>(*ctx);
        VPU::DynamicQuantizeOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::DynamicQuantizeOp>>(*ctx);
        VPU::PReluOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::PReluOp>>(*ctx);
        VPU::ExtractImagePatchesOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ExtractImagePatchesOp>>(*ctx);
        VPU::LeakyReluOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::LeakyReluOp>>(*ctx);
        VPU::MishOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::MishOp>>(*ctx);
        VPU::TileOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::TileOp>>(*ctx);
        VPU::ReLUOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ReLUOp>>(*ctx);
        VPU::YuvToRgbOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::YuvToRgbOp>>(*ctx);
        VPU::RandomUniformOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::RandomUniformOp>>(*ctx);
        VPU::OneHotOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::OneHotOp>>(*ctx);
        VPU::ReorgYoloOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ReorgYoloOp>>(*ctx);
        VPU::ProposalOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ProposalOp>>(*ctx);
        VPU::ScatterUpdateOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ScatterUpdateOp>>(*ctx);
        VPU::ScatterElementsUpdateOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ScatterElementsUpdateOp>>(
                *ctx);
        VPU::ReverseSequenceOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ReverseSequenceOp>>(*ctx);
        VPU::FloorModOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::FloorModOp>>(*ctx);
        VPU::ModOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ModOp>>(*ctx);
        VPU::EqualOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::EqualOp>>(*ctx);
        VPU::GreaterOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GreaterOp>>(*ctx);
        VPU::GreaterEqualOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GreaterEqualOp>>(*ctx);
        VPU::LessOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::LessOp>>(*ctx);
        VPU::LessEqualOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::LessEqualOp>>(*ctx);
        VPU::LogicalOrOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::LogicalOrOp>>(*ctx);
        VPU::HSigmoidOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::HSigmoidOp>>(*ctx);
        VPU::LogicalXorOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::LogicalXorOp>>(*ctx);
        VPU::LogicalNotOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::LogicalNotOp>>(*ctx);
        VPU::AndOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::AndOp>>(*ctx);
        VPU::NotEqualOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::NotEqualOp>>(*ctx);
        VPU::ReduceL1Op::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ReduceL1Op>>(*ctx);
        VPU::ReduceSumOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ReduceSumOp>>(*ctx);
        VPU::ReduceMeanOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ReduceMeanOp>>(*ctx);
        VPU::ReduceLogicalAndOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ReduceLogicalAndOp>>(*ctx);
        VPU::ReduceMaxOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ReduceMaxOp>>(*ctx);
        VPU::ReduceMinOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ReduceMinOp>>(*ctx);
        VPU::ReduceLogicalOrOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ReduceLogicalOrOp>>(*ctx);
        VPU::ReduceL2Op::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ReduceL2Op>>(*ctx);
        VPU::ReduceProdOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ReduceProdOp>>(*ctx);
        VPU::NegativeOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::NegativeOp>>(*ctx);
        VPU::NonMaxSuppressionOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::NonMaxSuppressionOp>>(*ctx);
        VPU::ROIPoolingOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ROIPoolingOp>>(*ctx);
        VPU::PSROIPoolingOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::PSROIPoolingOp>>(*ctx);
        VPU::PermuteQuantizeOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::PermuteQuantizeOp>>(*ctx);
        VPU::LogOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::LogOp>>(*ctx);
        VPU::FloorOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::FloorOp>>(*ctx);
        VPU::RoundOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::RoundOp>>(*ctx);
        VPU::SignOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SignOp>>(*ctx);
        VPU::SwishOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SwishOp>>(*ctx);
        VPU::SelectOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SelectOp>>(*ctx);
        VPU::EmbeddingBagOffsetsSumOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::EmbeddingBagOffsetsSumOp>>(
                *ctx);
        VPU::GRUSequenceOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GRUSequenceOp>>(*ctx);
        VPU::EmbeddingBagPackedSumOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::EmbeddingBagPackedSumOp>>(
                *ctx);
        VPU::GRUSequenceFirstPartOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GRUSequenceFirstPartOp>>(*ctx);
        VPU::GRUSequenceLastPartOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GRUSequenceLastPartOp>>(*ctx);
        VPU::LSTMCellOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::LSTMCellOp>>(*ctx);
        VPU::LSTMGatesOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::LSTMGatesOp>>(*ctx);
        VPU::LSTMSequenceOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::LSTMSequenceOp>>(*ctx);
        VPU::ErfOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ErfOp>>(*ctx);
        VPU::MaxPoolOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::MaxPoolOp>>(*ctx);
        VPU::RollOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::RollOp>>(*ctx);
        VPU::CTCGreedyDecoderSeqLenOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::CTCGreedyDecoderSeqLenOp>>(
                *ctx);
        VPU::AbsOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::AbsOp>>(*ctx);
        VPU::SquaredDifferenceOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SquaredDifferenceOp>>(*ctx);
        VPU::CTCGreedyDecoderOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::CTCGreedyDecoderOp>>(*ctx);
        VPU::GeluOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GeluOp>>(*ctx);
        VPU::SoftPlusOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SoftPlusOp>>(*ctx);
        VPU::ConvolutionOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ConvolutionOp>>(*ctx);
        VPU::GroupConvolutionOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GroupConvolutionOp>>(*ctx);
        VPU::DFTOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::DFTOp>>(*ctx);
        VPU::RDFTOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::RDFTOp>>(*ctx);
        VPU::IDFTOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::IDFTOp>>(*ctx);
        VPU::IRDFTOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::IRDFTOp>>(*ctx);
        VPU::RDFTUncutOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::RDFTUncutOp>>(*ctx);
        VPU::IRDFTLastAxisOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::IRDFTLastAxisOp>>(*ctx);
    });
}
