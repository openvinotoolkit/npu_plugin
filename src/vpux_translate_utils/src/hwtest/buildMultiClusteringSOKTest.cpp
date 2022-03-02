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

#include "vpux/compiler/core/passes.hpp"

namespace vpux {
namespace hwtest {

//
void buildMultiClusteringSOKTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
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
    const llvm::SmallVector<std::int64_t> weightsTableShape{weightsShape[0], 1, 1, 4};

    VPUX_THROW_UNLESS(!inputShape.empty(), "buildMultiClusteringSOK: Got empty inputShape");
    VPUX_THROW_UNLESS(!outputShape.empty(), "buildMultiClusteringSOK: Got empty outputShape");
    VPUX_THROW_UNLESS(!weightsShape.empty(), "buildMultiClusteringSOK: Got empty weightsShape");

    const char* weightsFileName = "weights.dat";
    auto numCluster = 4;
    VPUX_THROW_UNLESS(numCluster <= 4, "number of clustering must <= 4");

    auto ParentInputDistributed = VPUIP::DistributedBufferType::get(
            ctx, inputShape, inputType, mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx)),
            IndexedSymbolAttr::get(VPU::MemoryKindAttr::get(ctx, VPU::MemoryKind::CMX_NN)),
            VPU::DistributedTensorAttr::get(VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::DUPLICATED),
                                            nullptr, nullptr, VPU::getPaddingAttr(ctx, 0, 0, 0, 0), nullptr,
                                            vpux::getIntAttr(ctx, numCluster), ctx));

    auto ParentOutputDistributed = VPUIP::DistributedBufferType::get(
            ctx, outputShape, outputType, mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx)),
            IndexedSymbolAttr::get(VPU::MemoryKindAttr::get(ctx, VPU::MemoryKind::CMX_NN)),
            VPU::DistributedTensorAttr::get(
                    VPU::DistributionModeAttr::get(
                            ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED),
                    vpux::getIntArrayAttr(ctx, makeArrayRef({1, 2, 1, 1})), nullptr,
                    VPU::getPaddingAttr(ctx, 0, 0, 0, 0), nullptr, vpux::getIntAttr(ctx, numCluster), ctx));

    // get suboutputs and weight CMX shape
    const auto inputDistr = ParentInputDistributed.getDistribution();
    const auto outputDistr = ParentOutputDistributed.getDistribution();
    const auto inputMode = VPU::stringifyDistributionMode(inputDistr.mode().getValue());
    const auto outputMode = VPU::stringifyDistributionMode(outputDistr.mode().getValue());
    VPUX_THROW_UNLESS(inputMode == "DUPLICATED", "Input distribution mode must be DUPLICATED");
    VPUX_THROW_UNLESS(outputMode == "DUPLICATED|SEGMENTED", "Output distribution mode must be DUPLICATED|SEGMENTED");

    const auto ChannelsIndex = vpux::Dims4D::Act::C.ind();
    const auto inputChannels = inputShape[ChannelsIndex];
    const auto outputChannels = outputShape[ChannelsIndex];
    const auto alignmentRequirement = 16;
    VPUX_THROW_UNLESS(inputChannels % alignmentRequirement == 0, "Input channels must be multiple of {0}, got {1}",
                      alignmentRequirement, inputChannels);

    auto subOutputShape = outputShape;
    auto subWeightsCMXShape = weightsShape;
    auto subWeightsTableCMXShape = weightsTableShape;

    if (outputMode == "DUPLICATED|SEGMENTED") {
        VPUX_THROW_UNLESS(outputChannels % numCluster == 0,
                          "outputChannels must be multiple of number clusters {0}, got {1}", numCluster,
                          outputChannels);
        subOutputShape[ChannelsIndex] = subOutputShape[ChannelsIndex] / numCluster;
        subWeightsCMXShape[0] = subWeightsCMXShape[0] / numCluster;
        subWeightsTableCMXShape[0] = subWeightsTableCMXShape[0] / numCluster;
    }

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx));
    const auto elemStrides = SmallVector<int64_t>(
            {outputShape[1] * outputShape[2] * outputShape[3], 1, outputShape[1] * outputShape[3], outputShape[1]});
    const auto stridesAttr = getIntArrayAttr(ctx, elemStrides);
    const auto layout = IERT::MemRefAttr::get(orderAttr, stridesAttr, ctx);

    auto OutputDistributed = VPUIP::DistributedBufferType::get(
            ctx, subOutputShape, outputType, layout,
            IndexedSymbolAttr::get(VPU::MemoryKindAttr::get(ctx, VPU::MemoryKind::CMX_NN)),
            VPU::DistributedTensorAttr::get(VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::DUPLICATED),
                                            nullptr, nullptr, VPU::getPaddingAttr(ctx, 0, 0, 0, 0), nullptr,
                                            vpux::getIntAttr(ctx, numCluster), ctx));

    // get input, output, weight and weight table offset
    const auto inputCMXSize = vpux::hwtest::totalTensorSize(inputShape, inputType);
    const auto outputCMXSize = vpux::hwtest::totalTensorSize(outputShape, outputType);
    const auto subWeightsCMXSize = vpux::hwtest::totalTensorSize(subWeightsCMXShape, weightsType);

    const auto alignment =
            (alignmentRequirement * static_cast<vpux::Bit>(getElemTypeSize(inputType)).count()) / CHAR_BIT;

    const auto INPUT_CMX_OFFSET = 0;
    VPUX_THROW_UNLESS(INPUT_CMX_OFFSET % alignment == 0, "INPUT_CMX_OFFSET must be multiple of {0}, got {1}", alignment,
                      INPUT_CMX_OFFSET);

    const auto OUTPUT_CMX_OFFSET = INPUT_CMX_OFFSET + inputCMXSize;
    VPUX_THROW_UNLESS(OUTPUT_CMX_OFFSET % alignment == 0, "OUTPUT_CMX_OFFSET must be multiple of {0}, got {1}",
                      alignment, OUTPUT_CMX_OFFSET);

    const auto WEIGHTS_CMX_OFFSET = OUTPUT_CMX_OFFSET + outputCMXSize;
    VPUX_THROW_UNLESS(WEIGHTS_CMX_OFFSET % alignment == 0, "WEIGHTS_CMX_OFFSET must be multiple of {0}, got {1}",
                      alignment, WEIGHTS_CMX_OFFSET);

    const auto WEIGHTSTABLE_CMX_OFFSET = WEIGHTS_CMX_OFFSET + subWeightsCMXSize;
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
            llvm::formatv("multi_clustring_SOK_{0}_{1}_{2}", inputType, weightsType, outputType).str(), funcType,
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

    const auto MAX_DDR_USED_SIZE = OUTPUT_DDR_OFFSET + outputCMXSize;

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

    llvm::SmallVector<Const::DeclareOp> weightsDDRs, weightsTableDDRs;
    for (auto index = 0; index < numCluster; index++) {
        const auto offset = Shape({index * subWeightsCMXShape[0], 0, 0, 0});
        const auto shape = Shape(subWeightsCMXShape);
        const auto subWeightsAttribute = weightsAttribute.subview(offset, shape);

        const auto weightsDDRType =
                getMemRefType(VPURT::BufferSection::Constant, subWeightsCMXShape, weightsType, DimsOrder::NHWC)
                        .cast<vpux::NDTypeInterface>();
        weightsDDRs.push_back(functionBuilder.create<vpux::Const::DeclareOp>(builder.getUnknownLoc(), weightsDDRType,
                                                                             subWeightsAttribute));

        auto weightsStrides = weightsDDRType.getStrides();
        auto& weightsOutputChannelsStrideInBits = weightsStrides[vpux::Dims4D::Filter::OC];

        const auto weightsTableDDRType = mlir::RankedTensorType::get(subWeightsTableCMXShape, int32);
        const auto weightsTable = VPU::NCESparsity::getWeightsTable(
                inputType, outputType, static_cast<std::int32_t>(WEIGHTS_CMX_OFFSET),
                static_cast<std::int32_t>(weightsOutputChannelsStrideInBits.count() / CHAR_BIT),
                VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARISTY, vpux::VPU::ArchKind::KMB, subOutputShape[1],
                weightsType);

        const auto weightsTableDDRMemRef =
                getMemRefType(VPURT::BufferSection::Constant, subWeightsTableCMXShape, int32, DimsOrder::NCHW)
                        .cast<vpux::NDTypeInterface>();
        const auto weightsTableValues =
                mlir::DenseElementsAttr::get(weightsTableDDRType, llvm::makeArrayRef<std::int32_t>(weightsTable));
        weightsTableDDRs.push_back(functionBuilder.create<vpux::Const::DeclareOp>(
                builder.getUnknownLoc(), weightsTableDDRMemRef,
                vpux::Const::ContentAttr::get(weightsTableValues).reorder(vpux::DimsOrder::NCHW)));
    }

    llvm::SmallVector<int64_t> sectionIndex;
    for (auto index = 0; index < numCluster; index++) {
        sectionIndex.push_back(index);
    }

    // Define CMX buffer
    auto parentInputCMX = functionBuilder.create<VPURT::DeclareBufferOp>(
            builder.getUnknownLoc(), ParentInputDistributed, VPURT::BufferSection::CMX_NN, INPUT_CMX_OFFSET);
    auto parentInputCMXCopy = functionBuilder.create<VPURT::DeclareBufferOp>(
            builder.getUnknownLoc(), ParentInputDistributed, VPURT::BufferSection::CMX_NN, sectionIndex,
            INPUT_CMX_OFFSET);
    auto parentOutputCMX = functionBuilder.create<VPURT::DeclareBufferOp>(
            builder.getUnknownLoc(), ParentOutputDistributed, VPURT::BufferSection::CMX_NN, OUTPUT_CMX_OFFSET);
    auto parentOutputCMXCompact = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, outputShape,
                                                        outputType, vpux::DimsOrder::NHWC, 0, OUTPUT_CMX_OFFSET);

    llvm::SmallVector<VPURT::DeclareBufferOp> subInputCMX;
    llvm::SmallVector<VPURT::DeclareBufferOp> subOutputCMX;
    llvm::SmallVector<VPURT::DeclareBufferOp> subWeightsCMX;
    llvm::SmallVector<VPURT::DeclareBufferOp> subWeightsTableCMX;

    for (auto index = 0; index < numCluster; index++) {
        subInputCMX.push_back(createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, inputShape,
                                                    inputType, vpux::DimsOrder::NHWC, index, INPUT_CMX_OFFSET));
        subOutputCMX.push_back(functionBuilder.create<VPURT::DeclareBufferOp>(
                builder.getUnknownLoc(), OutputDistributed, VPURT::BufferSection::CMX_NN, sectionIndex,
                OUTPUT_CMX_OFFSET));
        subWeightsCMX.push_back(createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, subWeightsCMXShape,
                                                      weightsType, vpux::DimsOrder::NHWC, index, WEIGHTS_CMX_OFFSET));
        subWeightsTableCMX.push_back(createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN,
                                                           subWeightsTableCMXShape, int32, DimsOrder::NCHW, index,
                                                           WEIGHTSTABLE_CMX_OFFSET));
    }

    int barrierNumber = 0;

    // Reorder input / copy input from inputbuffer to DDR
    auto updateBarrier =
            functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);
    //     const auto inputMemPerm = vpux::getPermutationFromOrders(DimsOrder::NCHW, DimsOrder::NHWC, ctx);
    //     VPURT::wrapIntoTaskOp<VPUIP::PermuteUPAOp>(
    //             functionBuilder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.barrier()),
    //             builder.getUnknownLoc(), functionInput, parentInputDDR.getOperation()->getResult(0), inputMemPerm);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(), updateBarrier.barrier(),
                                          builder.getUnknownLoc(), functionInput,
                                          parentInputDDR.getOperation()->getResult(0));
    VPURT::ConfigureBarrierOp waitInputReorderBarrier = updateBarrier;

    // Copy input from DDR to CMX
    updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, waitInputReorderBarrier.barrier(), updateBarrier.barrier(),
                                          builder.getUnknownLoc(), parentInputDDR.getOperation()->getResult(0),
                                          parentInputCMXCopy.getOperation()->getResult(0));

    VPURT::ConfigureBarrierOp waitInputToCMXBarrier = updateBarrier;

    // Move weights and weights table from DDR to CMX
    updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);

    for (auto index = 0; index < numCluster; index++) {
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                functionBuilder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.barrier()), builder.getUnknownLoc(),
                weightsDDRs[index].getOperation()->getResult(0), subWeightsCMX[index].getOperation()->getResult(0));
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(),
                                              mlir::ValueRange(updateBarrier.barrier()), builder.getUnknownLoc(),
                                              weightsTableDDRs[index].getOperation()->getResult(0),
                                              subWeightsTableCMX[index].getOperation()->getResult(0));
    }

    VPURT::ConfigureBarrierOp waitWeightsBarrier = updateBarrier;

    // tile task
    updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);

    const auto strides = getIntArrayAttr(ctx, conv.stride);
    std::vector<std::int64_t> paddings = convertNBPadtoNCETaskPad(conv.pad);
    const auto kernelPaddings = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                                    paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);
    llvm::SmallVector<std::int64_t> kernel = {weightsShape[2], weightsShape[3]};
    const auto kernelSize = getIntArrayAttr(ctx, kernel);

    for (auto index = 0; index < numCluster; index++) {
        auto nceTask = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
                functionBuilder, mlir::ValueRange({waitWeightsBarrier.barrier(), waitInputToCMXBarrier.barrier()}),
                updateBarrier.barrier(), builder.getUnknownLoc(), subInputCMX[index].getOperation()->getResult(0),
                subWeightsCMX[index].getOperation()->getResult(0),
                subWeightsTableCMX[index].getOperation()->getResult(0), nullptr,
                parentInputCMX.getOperation()->getResult(0), parentOutputCMX.getOperation()->getResult(0),
                subOutputCMX[index].getOperation()->getResult(0), vpux::VPUIP::NCETaskType::CONV, kernelSize, strides,
                kernelPaddings, nullptr, nullptr);

        const auto start = getIntArrayAttr(ctx, std::vector<std::int64_t>{0, 0, inputShape[ChannelsIndex] * index});
        const auto end = getIntArrayAttr(ctx, std::vector<std::int64_t>{outputShape[3] - 1, outputShape[2] - 1,
                                                                        inputShape[1] * (index + 1) - 1});
        const auto pad = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                             paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);

        nceTask.addDPUTask(functionBuilder, start, end, pad, conv.cube_mode);
    }

    VPURT::ConfigureBarrierOp waitDPUTaskBarrier = updateBarrier;

    // copy output from CMX to DDR
    updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, waitDPUTaskBarrier.barrier(), updateBarrier.barrier(),
                                          builder.getUnknownLoc(), parentOutputCMXCompact.getOperation()->getResult(0),
                                          parentOutputDDR.getOperation()->getResult(0));

    // reorder output / copy output data from DDR to output buffer
    VPURT::ConfigureBarrierOp waitCMXToDDRBarrier = updateBarrier;
    //     const auto outMemPerm = vpux::getPermutationFromOrders(DimsOrder::NHWC, DimsOrder::NCHW, ctx);
    //     VPURT::wrapIntoTaskOp<VPUIP::PermuteUPAOp>(functionBuilder, waitCMXToDDRBarrier.barrier(),
    //     mlir::ValueRange(),
    //                                                builder.getUnknownLoc(),
    //                                                parentOutputDDR.getOperation()->getResult(0), functionOutput,
    //                                                outMemPerm);
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

    // output dot
    pm.addPass(
            vpux::createPrintDotPass("/home/sgl/Github/openvino/bin/intel64/Release/dot/PSS.dot", {}, {}, true, true));

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
