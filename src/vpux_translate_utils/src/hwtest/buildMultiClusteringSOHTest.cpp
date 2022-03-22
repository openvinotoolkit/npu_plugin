//
// Copyright 2022 Intel Corporation.
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

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/numeric.hpp"

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"

namespace vpux {
namespace hwtest {

//
void buildMultiClusteringSOHTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                                 mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type weightsType,
                                 mlir::Type outputType) {
    auto* ctx = builder.getContext();
    const auto int32 = builder.getIntegerType(32, true);

    const auto input = testDesc.getInputLayer();
    const auto weights = testDesc.getWeightLayer();
    const auto conv = testDesc.getConvLayer();
    const auto output = testDesc.getOutputLayer();

    const llvm::SmallVector<std::int64_t> inputShape(input.shape.begin(), input.shape.end());
    const llvm::SmallVector<std::int64_t> outputShape(output.shape.begin(), output.shape.end());
    const llvm::SmallVector<std::int64_t> weightsShape{weights.shape.begin(), weights.shape.end()};
    const llvm::SmallVector<std::int64_t> weightsTableShape{weightsShape[vpux::Dims4D::Filter::OC.ind()], 1, 1, 4};

    VPUX_THROW_UNLESS(!inputShape.empty(), "buildMultiClustering: Got empty inputShape");
    VPUX_THROW_UNLESS(!outputShape.empty(), "buildMultiClustering: Got empty outputShape");
    VPUX_THROW_UNLESS(!weightsShape.empty(), "buildMultiClustering: Got empty weightsShape");

    const char* weightsFileName = "weights.dat";
    const auto numCluster = testDesc.getClusterNumber();
    VPUX_THROW_UNLESS(numCluster <= 4 && numCluster > 1, "number of clustering must (1, 4], but got {0}", numCluster);

    // Define activation, output, weight and weightTable distribution type
    auto NCHW = mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(ctx));
    auto NHWC = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx));
    auto locateAtCMX = IndexedSymbolAttr::get(VPU::MemoryKindAttr::get(ctx, VPU::MemoryKind::CMX_NN));

    auto activationMode = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::SEGMENTED);
    auto outputMode = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::SEGMENTED);
    auto weightsMode = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::DUPLICATED);
    auto weightsTableMode = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::DUPLICATED);

    if (weightsShape[Dims4D::Filter::KX.ind()] > 1 || weightsShape[Dims4D::Filter::KY.ind()] > 1 ||
        conv.stride[0] > 1 || conv.stride[1] > 1) {
        activationMode = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::OVERLAPPED);
    }

    auto numTilesAttr = getIntArrayAttr(ctx, makeArrayRef({1, 1, checked_cast<int32_t>(numCluster), 1}));
    auto kernelAttr = getIntArrayAttr(
            ctx, makeArrayRef({weightsShape[Dims4D::Filter::KY.ind()], weightsShape[Dims4D::Filter::KX.ind()]}));
    std::vector<int64_t> paddings = convertNBPadtoNCETaskPad(conv.pad);
    auto padAttr = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                       paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);
    auto stridesAttr = getIntArrayAttr(ctx, conv.stride);
    auto numClusterAttr = getIntAttr(ctx, checked_cast<int32_t>(numCluster));
    auto alignmentAttr = nullptr;

    auto inputDistribute = VPUIP::DistributedBufferType::get(
            ctx, inputShape, inputType, NHWC, locateAtCMX,
            VPU::DistributedTensorAttr::get(activationMode, numTilesAttr, kernelAttr, padAttr, stridesAttr,
                                            numClusterAttr, alignmentAttr, ctx));

    auto outputDistribute = VPUIP::DistributedBufferType::get(
            ctx, outputShape, outputType, NHWC, locateAtCMX,
            VPU::DistributedTensorAttr::get(outputMode, numTilesAttr, kernelAttr, padAttr, stridesAttr, numClusterAttr,
                                            alignmentAttr, ctx));

    auto weightsDistribute = VPUIP::DistributedBufferType::get(
            ctx, weightsShape, weightsType, NHWC, locateAtCMX,
            VPU::DistributedTensorAttr::get(weightsMode, numTilesAttr, kernelAttr, padAttr, stridesAttr, numClusterAttr,
                                            alignmentAttr, ctx));

    auto weightsTableDistribute = VPUIP::DistributedBufferType::get(
            ctx, weightsTableShape, int32, NCHW, locateAtCMX,
            VPU::DistributedTensorAttr::get(weightsTableMode, numTilesAttr, kernelAttr, padAttr, stridesAttr,
                                            numClusterAttr, alignmentAttr, ctx));

    // get input, output and weight CMX shape
    const auto inputChannels = inputShape[Dims4D::Act::C.ind()];
    const auto outputHeight = outputShape[Dims4D::Act::H.ind()];
    const auto alignmentRequirement = 16;
    VPUX_THROW_UNLESS(inputChannels % alignmentRequirement == 0, "input channels must be multiple of {0}, got {1}",
                      alignmentRequirement, inputChannels);
    // Each DPU should compute at least one output line. Therefore in order for a layer to be SOH
    // compitable it must have an output height of at least the number of DPUs x the number of clusters
    VPUX_THROW_UNLESS(outputHeight >= checked_cast<int64_t>(numCluster) * KMB_PER_CLUSTER_DPU_NUMBER,
                      "outputHeight must be larger than DPU number {0}, got {1}", outputHeight,
                      checked_cast<int64_t>(numCluster) * KMB_PER_CLUSTER_DPU_NUMBER);

    // get per cluster shape
    auto subInputCMXShapes = inputDistribute.getPerClusterComputeShapes();
    auto subOutputCMXShapes = outputDistribute.getPerClusterComputeShapes();
    auto weightsCMXShape = weightsDistribute.getPerClusterComputeShapes();
    VPUX_THROW_UNLESS(subInputCMXShapes.size() == numCluster && subOutputCMXShapes.size() == numCluster &&
                              weightsCMXShape.size() == numCluster,
                      "Sub Shape number must be same with number cluster");

    // get input, output, weight and weight table offset
    // Difference cluster maybe has difference input size, and the first one is always largest.
    // So, we just simply used first cluster to calculate offsets.
    const auto subInputCMXSize = vpux::hwtest::totalTensorSize(subInputCMXShapes.front().totalSize(), inputType);
    const auto subOutputCMXSize = vpux::hwtest::totalTensorSize(subOutputCMXShapes.front().totalSize(), outputType);
    const auto weightsCMXSize = vpux::hwtest::totalTensorSize(weightsCMXShape.front().totalSize(), weightsType);

    const auto alignment =
            (alignmentRequirement * static_cast<vpux::Bit>(getElemTypeSize(inputType)).count()) / CHAR_BIT;

    const auto INPUT_CMX_OFFSET = 0;
    VPUX_THROW_UNLESS(INPUT_CMX_OFFSET % alignment == 0, "INPUT_CMX_OFFSET must be multiple of {0}, got {1}", alignment,
                      INPUT_CMX_OFFSET);

    const auto OUTPUT_CMX_OFFSET = INPUT_CMX_OFFSET + subInputCMXSize;
    VPUX_THROW_UNLESS(OUTPUT_CMX_OFFSET % alignment == 0, "OUTPUT_CMX_OFFSET must be multiple of {0}, got {1}",
                      alignment, OUTPUT_CMX_OFFSET);

    const auto WEIGHTS_CMX_OFFSET = OUTPUT_CMX_OFFSET + subOutputCMXSize;
    VPUX_THROW_UNLESS(WEIGHTS_CMX_OFFSET % alignment == 0, "WEIGHTS_CMX_OFFSET must be multiple of {0}, got {1}",
                      alignment, WEIGHTS_CMX_OFFSET);

    const auto WEIGHTSTABLE_CMX_OFFSET = WEIGHTS_CMX_OFFSET + weightsCMXSize;
    VPUX_THROW_UNLESS(WEIGHTSTABLE_CMX_OFFSET % alignment == 0,
                      "WEIGHTSTABLE_CMX_OFFSET must be multiple of {0}, got {1}", alignment, WEIGHTSTABLE_CMX_OFFSET);

    // define function
    const auto outputParamType =
            getMemRefType(VPURT::BufferSection::NetworkOutput, outputShape, outputType, DimsOrder::NHWC)
                    .cast<vpux::NDTypeInterface>();
    const auto inputParamType =
            getMemRefType(VPURT::BufferSection::NetworkInput, inputShape, inputType, DimsOrder::NHWC)
                    .cast<vpux::NDTypeInterface>();

    const auto funcType =
            builder.getFunctionType(SmallVector<mlir::Type>{inputParamType, outputParamType}, outputParamType);

    auto function = builder.create<mlir::FuncOp>(
            builder.getUnknownLoc(),
            llvm::formatv("multi_clustering_SOH_{0}_{1}_{2}", inputType, weightsType, outputType).str(), funcType,
            builder.getStringAttr("private"));

    auto functionBuilder = mlir::OpBuilder::atBlockBegin(function.addEntryBlock(), builder.getListener());

    auto functionInput = function.getArgument(0);
    auto functionOutput = function.getArgument(1);

    // Define input and output DDR buffer
    const auto inputDDRSize = vpux::hwtest::totalTensorSize(inputShape, inputType);
    const auto INPUT_DDR_OFFSET = 0;
    VPUX_THROW_UNLESS(INPUT_DDR_OFFSET % alignment == 0, "INPUT_DDR_OFFSET must be multiple of {0}, got {1}", alignment,
                      INPUT_DDR_OFFSET);

    const auto OUTPUT_DDR_OFFSET = INPUT_DDR_OFFSET + inputDDRSize;
    VPUX_THROW_UNLESS(OUTPUT_DDR_OFFSET % alignment == 0, "OUTPUT_DDR_OFFSET must be multiple of {0}, got {1}",
                      alignment, OUTPUT_DDR_OFFSET);

    auto parentInputDDR = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::DDR, inputShape, inputType,
                                                vpux::DimsOrder::NHWC, INPUT_DDR_OFFSET);
    auto parentOutputDDR = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::DDR, outputShape, outputType,
                                                 vpux::DimsOrder::NHWC, OUTPUT_DDR_OFFSET);

    llvm::SmallVector<VPURT::DeclareBufferOp> subInputDDRs;
    llvm::SmallVector<VPURT::DeclareBufferOp> subOutputDDRs;
    const auto perClusterShapeOffsets = inputDistribute.getPerClusterComputeShapeOffsets();
    auto INPUT_WC_STRIDE = vpux::hwtest::totalTensorSize(inputShape, inputType) / inputShape[Dims4D::Act::H.ind()];
    auto SUBINPUT_DDR_OFFSET = INPUT_DDR_OFFSET;
    auto SUBOUTPUT_DDR_OFFSET = OUTPUT_DDR_OFFSET;
    for (std::size_t index = 0; index < numCluster; index++) {
        SUBINPUT_DDR_OFFSET = INPUT_DDR_OFFSET + perClusterShapeOffsets[index][Dims4D::Act::H] * INPUT_WC_STRIDE;
        subInputDDRs.push_back(createDeclareTensorOp(functionBuilder, VPURT::BufferSection::DDR,
                                                     subInputCMXShapes[index].raw(), inputType, vpux::DimsOrder::NHWC,
                                                     SUBINPUT_DDR_OFFSET));
        subOutputDDRs.push_back(createDeclareTensorOp(functionBuilder, VPURT::BufferSection::DDR,
                                                      subOutputCMXShapes[index].raw(), outputType,
                                                      vpux::DimsOrder::NHWC, SUBOUTPUT_DDR_OFFSET));
        SUBOUTPUT_DDR_OFFSET += vpux::hwtest::totalTensorSize(subOutputCMXShapes[index].totalSize(), outputType);
    }

    const auto OUTPUT_DDR_SIZE = vpux::hwtest::totalTensorSize(outputShape, outputType);
    const auto MAX_DDR_USED_SIZE = OUTPUT_DDR_OFFSET + OUTPUT_DDR_SIZE;

    // Define weights and weights table DDR buffer
    const auto weightsValues = generateWeights(weightsShape, weightsType, ctx, weightsFileName);
    auto weightsAttribute = vpux::Const::ContentAttr::get(weightsValues);
    weightsAttribute = weightsAttribute.reorder(vpux::DimsOrder::OYXI);

    auto qty = weightsType.dyn_cast<mlir::quant::QuantizedType>();

    if (qty != nullptr) {
        if (qty.getStorageType().isInteger(4)) {
            weightsAttribute = weightsAttribute.bitPack(4);
        }
        weightsAttribute = weightsAttribute.quantCast(qty);
    }

    const auto weightsDDRType =
            getMemRefType(VPURT::BufferSection::Constant, weightsShape, weightsType, DimsOrder::NHWC)
                    .cast<vpux::NDTypeInterface>();
    auto weightsDDR =
            functionBuilder.create<vpux::Const::DeclareOp>(builder.getUnknownLoc(), weightsDDRType, weightsAttribute);

    auto weightsStrides = weightsDDRType.getStrides();
    auto& weightsOutputChannelsStrideInBits = weightsStrides[vpux::Dims4D::Filter::OC];

    const auto weightsTableDDRType = mlir::RankedTensorType::get(weightsTableShape, int32);
    const auto weightsTable = VPU::NCESparsity::getWeightsTable(
            inputType, outputType, static_cast<std::int32_t>(WEIGHTS_CMX_OFFSET),
            static_cast<std::int32_t>(weightsOutputChannelsStrideInBits.count() / CHAR_BIT),
            VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY, vpux::VPU::ArchKind::KMB, output.shape[1], weightsType);

    const auto weightsTableDDRMemRef =
            getMemRefType(VPURT::BufferSection::Constant, weightsTableShape, int32, DimsOrder::NCHW)
                    .cast<vpux::NDTypeInterface>();
    const auto weightsTableValues =
            mlir::DenseElementsAttr::get(weightsTableDDRType, llvm::makeArrayRef<std::int32_t>(weightsTable));
    auto weightsTableDDR = functionBuilder.create<vpux::Const::DeclareOp>(
            builder.getUnknownLoc(), weightsTableDDRMemRef,
            vpux::Const::ContentAttr::get(weightsTableValues).reorder(vpux::DimsOrder::NCHW));

    // Define CMX buffer
    llvm::SmallVector<int64_t> sectionIndex;
    for (std::size_t index = 0; index < numCluster; index++) {
        sectionIndex.push_back(index);
    }

    auto parentInputCMX =
            createDeclareTensorOp(functionBuilder, inputDistribute, VPURT::BufferSection::CMX_NN, INPUT_CMX_OFFSET);
    auto parentOutputCMX =
            createDeclareTensorOp(functionBuilder, outputDistribute, VPURT::BufferSection::CMX_NN, OUTPUT_CMX_OFFSET);
    auto parentWeightsCMX = createDeclareTensorOp(functionBuilder, weightsDistribute, VPURT::BufferSection::CMX_NN,
                                                  sectionIndex, WEIGHTS_CMX_OFFSET);
    auto parentWeightsTableCMX =
            createDeclareTensorOp(functionBuilder, weightsTableDistribute, VPURT::BufferSection::CMX_NN, sectionIndex,
                                  WEIGHTSTABLE_CMX_OFFSET);

    llvm::SmallVector<VPURT::DeclareBufferOp> subInputCMX;
    llvm::SmallVector<VPURT::DeclareBufferOp> subOutputCMX;
    llvm::SmallVector<VPURT::DeclareBufferOp> subWeightsCMX;
    llvm::SmallVector<VPURT::DeclareBufferOp> subWeightsTableCMX;

    for (std::size_t index = 0; index < numCluster; index++) {
        subInputCMX.push_back(createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN,
                                                    subInputCMXShapes[index].raw(), inputType, vpux::DimsOrder::NHWC,
                                                    index, INPUT_CMX_OFFSET));
        subOutputCMX.push_back(createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN,
                                                     subOutputCMXShapes[index].raw(), outputType, vpux::DimsOrder::NHWC,
                                                     index, OUTPUT_CMX_OFFSET));
        subWeightsCMX.push_back(createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN,
                                                      weightsCMXShape[index].raw(), weightsType, vpux::DimsOrder::NHWC,
                                                      index, WEIGHTS_CMX_OFFSET));
        subWeightsTableCMX.push_back(createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN,
                                                           weightsTableShape, int32, DimsOrder::NCHW, index,
                                                           WEIGHTSTABLE_CMX_OFFSET));
    }

    int barrierNumber = 0;
    auto createBarrier = [&]() -> VPURT::ConfigureBarrierOp {
        barrierNumber = barrierNumber > 31 ? 1 : barrierNumber;
        return functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);
    };

    // Step1: Copy input from inputbuffer to DDR
    auto updateBarrier = createBarrier();

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(), updateBarrier.barrier(),
                                          builder.getUnknownLoc(), functionInput,
                                          parentInputDDR.getOperation()->getResult(0));
    VPURT::ConfigureBarrierOp waitInputReorderBarrier = updateBarrier;

    // Step2: Copy input from DDR to CMX
    updateBarrier = createBarrier();
    for (std::size_t index = 0; index < numCluster; index++) {
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                functionBuilder, waitInputReorderBarrier.barrier(), updateBarrier.barrier(), builder.getUnknownLoc(),
                subInputDDRs[index].getOperation()->getResult(0), subInputCMX[index].getOperation()->getResult(0));
    }
    VPURT::ConfigureBarrierOp waitInputToCMXBarrier = updateBarrier;

    // Step3: Move weights and weights table from DDR to CMX
    updateBarrier = createBarrier();

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
            functionBuilder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.barrier()), builder.getUnknownLoc(),
            weightsDDR.getOperation()->getResult(0), parentWeightsCMX.getOperation()->getResult(0));
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
            functionBuilder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.barrier()), builder.getUnknownLoc(),
            weightsTableDDR.getOperation()->getResult(0), parentWeightsTableCMX.getOperation()->getResult(0));

    VPURT::ConfigureBarrierOp waitWeightsBarrier = updateBarrier;

    // Step4: Sub NCE task
    updateBarrier = createBarrier();

    auto heightOffset = 0;
    for (std::size_t clusterIndex = 0; clusterIndex < numCluster; clusterIndex++) {
        auto nceTask = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
                functionBuilder, mlir::ValueRange({waitWeightsBarrier.barrier(), waitInputToCMXBarrier.barrier()}),
                updateBarrier.barrier(), builder.getUnknownLoc(),
                subInputCMX[clusterIndex].getOperation()->getResult(0),
                subWeightsCMX[clusterIndex].getOperation()->getResult(0),
                subWeightsTableCMX[clusterIndex].getOperation()->getResult(0), nullptr,
                parentInputCMX.getOperation()->getResult(0), parentOutputCMX.getOperation()->getResult(0),
                subOutputCMX[clusterIndex].getOperation()->getResult(0), vpux::VPUIP::NCETaskType::CONV, kernelAttr,
                stridesAttr, padAttr, nullptr, nullptr);

        // Because we have't workload cost model, Just split in C dim
        auto numOfDPUTask = subOutputCMXShapes[clusterIndex][vpux::Dims4D::Act::C] / alignmentRequirement;
        VPUX_THROW_UNLESS(numOfDPUTask <= KMB_PER_CLUSTER_DPU_NUMBER,
                          "DPU task number must be less than DPU number {0}, got {1}", numOfDPUTask,
                          KMB_PER_CLUSTER_DPU_NUMBER);

        // For SOH pad value need recalculate for each DPU task
        auto padOfDPUTask = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT], 0, 0);

        if (numCluster == 1) {
            padOfDPUTask = padAttr;
        }
        if (numCluster > 1 && clusterIndex == 0) {
            padOfDPUTask = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                               paddings[PAD_NCETASK_TOP], 0);
        }
        if (numCluster > 1 && clusterIndex == numCluster - 1) {
            padOfDPUTask = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT], 0,
                                               paddings[PAD_NCETASK_BOTTOM]);
        }

        for (auto dpuIndex = 0; dpuIndex < numOfDPUTask; dpuIndex++) {
            const auto start =
                    getIntArrayAttr(ctx, std::vector<std::int64_t>{0, heightOffset, dpuIndex * alignmentRequirement});
            auto end = getIntArrayAttr(
                    ctx,
                    std::vector<std::int64_t>{outputShape[vpux::Dims4D::Act::W.ind()] - 1,
                                              heightOffset + subOutputCMXShapes[clusterIndex][vpux::Dims4D::Act::H] - 1,
                                              (dpuIndex + 1) * alignmentRequirement - 1});

            nceTask.addDPUTask(functionBuilder, start, end, padOfDPUTask, conv.cube_mode);
        }

        heightOffset += subOutputCMXShapes[clusterIndex][vpux::Dims4D::Act::H];
    }

    VPURT::ConfigureBarrierOp waitDPUTaskBarrier = updateBarrier;

    // Step5: Copy output from CMX to DDR
    updateBarrier = createBarrier();
    for (std::size_t index = 0; index < numCluster; index++) {
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, waitDPUTaskBarrier.barrier(), updateBarrier.barrier(),
                                              builder.getUnknownLoc(), subOutputCMX[index].getOperation()->getResult(0),
                                              subOutputDDRs[index].getOperation()->getResult(0));
    }

    // Step6: Copy output data from DDR to output buffer
    VPURT::ConfigureBarrierOp waitCMXToDDRBarrier = updateBarrier;

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, waitCMXToDDRBarrier.barrier(), mlir::ValueRange(),
                                          builder.getUnknownLoc(), parentOutputDDR.getOperation()->getResult(0),
                                          functionOutput);

    functionBuilder.create<mlir::ReturnOp>(builder.getUnknownLoc(), functionOutput);

    module.dump();

    mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPU::createInitCompilerPass(VPU::ArchKind::KMB, VPU::CompilationMode::DefaultHW, None, log));
    if (conv.compress) {
        pm.addPass(VPUIP::createCompressWeightsPass(log));
    }

    // set memory sizes
    auto usedMemModule = module.lookupSymbol<mlir::ModuleOp>(IE::usedMemModuleName);
    if (usedMemModule == nullptr) {
        usedMemModule = builder.create<mlir::ModuleOp>(module->getLoc(), IE::usedMemModuleName);
    }
    IE::addAvailableMemory(usedMemModule, VPU::MemoryKind::CMX_NN, VPU::KMB_CMX_WORKSPACE_SIZE);
    IE::addAvailableMemory(usedMemModule, VPU::MemoryKind::DDR, Byte(MAX_DDR_USED_SIZE));

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    buildCNNOp(builder, function.getName(),
               {getTensorType(ShapeRef(inputShape), inputType, vpux::DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(outputShape), outputType, vpux::DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
