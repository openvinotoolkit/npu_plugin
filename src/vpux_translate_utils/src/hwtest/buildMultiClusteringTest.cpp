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

namespace vpux {
namespace hwtest {

//

// void buildMultiClustering(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
//                           Logger& log, VPUIP::DistributedBufferType inputDistribute,
//                           VPUIP::DistributedBufferType weightsDistribute,
//                           VPUIP::DistributedBufferType weightsTableDistribute,
//                           VPUIP::DistributedBufferType outputDistribute) {
void buildMultiClustering(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                          Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType) {
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

    VPUX_THROW_UNLESS(!inputShape.empty(), "buildMultiClustering: Got empty inputShape");
    VPUX_THROW_UNLESS(!outputShape.empty(), "buildMultiClustering: Got empty outputShape");
    VPUX_THROW_UNLESS(!weightsShape.empty(), "buildMultiClustering: Got empty weightsShape");

    const char* weightsFileName = "weights.dat";

    auto numCluster = 2;
    llvm::SmallVector<int64_t> sectionIndex{0, 1};

    auto inputDistribute = VPUIP::DistributedBufferType::get(
            ctx, inputShape, inputType, mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx)),
            IndexedSymbolAttr::get(VPU::MemoryKindAttr::get(ctx, VPU::MemoryKind::CMX_NN)),
            VPU::DistributedTensorAttr::get(VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::SEGMENTED),
                                            nullptr, nullptr, VPU::getPaddingAttr(ctx, 0, 0, 0, 0), nullptr,
                                            vpux::getIntAttr(ctx, numCluster), ctx));

    auto outputDistribute = VPUIP::DistributedBufferType::get(
            ctx, outputShape, outputType, mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx)),
            IndexedSymbolAttr::get(VPU::MemoryKindAttr::get(ctx, VPU::MemoryKind::CMX_NN)),
            VPU::DistributedTensorAttr::get(VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::SEGMENTED),
                                            nullptr, nullptr, VPU::getPaddingAttr(ctx, 0, 0, 0, 0), nullptr,
                                            vpux::getIntAttr(ctx, numCluster), ctx));

    auto weightsDistribute = VPUIP::DistributedBufferType::get(
            ctx, weightsShape, weightsType, mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx)),
            IndexedSymbolAttr::get(VPU::MemoryKindAttr::get(ctx, VPU::MemoryKind::CMX_NN)),
            VPU::DistributedTensorAttr::get(VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::DUPLICATED),
                                            nullptr, nullptr, VPU::getPaddingAttr(ctx, 0, 0, 0, 0), nullptr, nullptr,
                                            ctx));

    auto weightsTableDistribute = VPUIP::DistributedBufferType::get(
            ctx, weightsTableShape, int32, mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(ctx)),
            IndexedSymbolAttr::get(VPU::MemoryKindAttr::get(ctx, VPU::MemoryKind::CMX_NN)),
            VPU::DistributedTensorAttr::get(VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::DUPLICATED),
                                            nullptr, nullptr, VPU::getPaddingAttr(ctx, 0, 0, 0, 0), nullptr, nullptr,
                                            ctx));

    // get input, output and weight CMX shape
    const auto inputDistr = inputDistribute.getDistribution();
    const auto outputDistr = outputDistribute.getDistribution();
    const auto inputMode = VPU::stringifyDistributionMode(inputDistr.mode().getValue());
    const auto outputMode = VPU::stringifyDistributionMode(outputDistr.mode().getValue());
    VPUX_THROW_UNLESS(inputMode == "SEGMENTED" || inputMode == "DUPLICATED",
                      "input distribution mode must be SEGMENTED or DUPLICATED");
    VPUX_THROW_UNLESS(outputMode == inputMode, "Input mode must same with output mode for SOH");

    const auto ChannelsIndex = vpux::Dims4D::Act::C.ind();
    const auto HeightIndex = vpux::Dims4D::Act::H.ind();
    const auto inputChannels = inputShape[ChannelsIndex];
    const auto inputHeight = inputShape[HeightIndex];
    const auto outputHeight = outputShape[HeightIndex];
    const auto alignmentRequirement = 16;
    VPUX_THROW_UNLESS(inputChannels % alignmentRequirement == 0, "input channels must be multiple of {0}, got {1}",
                      alignmentRequirement, inputChannels);

    auto subInputShape = inputShape;
    auto subOutputShape = outputShape;

    if (inputMode == "SEGMENTED") {
        numCluster = inputDistr.num_clusters().getInt();
        VPUX_THROW_UNLESS(inputHeight % numCluster == 0,
                          "inputChannels must be multiple of number clusters {0}, got {1}", numCluster, inputHeight);
        VPUX_THROW_UNLESS(outputHeight % numCluster == 0,
                          "outputChannels must be multiple of number clusters {0}, got {1}", numCluster, outputHeight);
        subInputShape[HeightIndex] = subInputShape[HeightIndex] / numCluster;
        subOutputShape[HeightIndex] = subOutputShape[HeightIndex] / numCluster;
    }

    const auto subInputShapes = llvm::SmallVector<llvm::SmallVector<std::int64_t>>(numCluster, subInputShape);
    const auto subOutputShapes = llvm::SmallVector<llvm::SmallVector<std::int64_t>>(numCluster, subOutputShape);

    const auto weightsDistr = weightsDistribute.getDistribution();
    const auto weightsMode = VPU::stringifyDistributionMode(weightsDistr.mode().getValue());
    VPUX_THROW_UNLESS(weightsMode == "DUPLICATED", "weights distribution mode must be DUPLICATED");
    auto weightsCMXShape = weightsShape;

    // get input, output, weight and weight table offset
    //     const auto inputType = inputDistribute.getElementType();
    //     const auto outputType = outputDistribute.getElementType();
    //     const auto weightsType = weightsDistribute.getElementType();

    const auto subInputCMXSize = vpux::hwtest::totalTensorSize(subInputShapes.back(), inputType);
    const auto subOutputCMXSize = vpux::hwtest::totalTensorSize(subOutputShapes.back(), outputType);
    const auto weightsCMXSize = vpux::hwtest::totalTensorSize(weightsCMXShape, weightsType);

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
            getMemRefType(VPURT::BufferSection::NetworkOutput, outputShape, outputType, DimsOrder::NCHW)
                    .cast<vpux::NDTypeInterface>();
    const auto inputParamType =
            getMemRefType(VPURT::BufferSection::NetworkInput, inputShape, inputType, DimsOrder::NCHW)
                    .cast<vpux::NDTypeInterface>();

    const auto funcType =
            builder.getFunctionType(SmallVector<mlir::Type>{inputParamType, outputParamType}, outputParamType);

    auto function = builder.create<mlir::FuncOp>(
            builder.getUnknownLoc(),
            llvm::formatv("multi_clustring_{0}_{1}_{2}", inputType, weightsType, outputType).str(), funcType,
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

    const auto INPUT_DDR_STRIDE = vpux::hwtest::totalTensorSize(subInputShapes.back(), inputType);
    const auto OUTPUT_DDR_STRIDE = vpux::hwtest::totalTensorSize(subOutputShapes.back(), inputType);

    auto parentInputDDR = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::DDR, inputShape, inputType,
                                                vpux::DimsOrder::NHWC, INPUT_DDR_OFFSET);
    auto parentOutputDDR = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::DDR, outputShape, outputType,
                                                 vpux::DimsOrder::NHWC, OUTPUT_DDR_OFFSET);

    llvm::SmallVector<VPURT::DeclareBufferOp> subInputDDR;
    llvm::SmallVector<VPURT::DeclareBufferOp> subOutputDDR;
    for (auto index = 0; index < numCluster; index++) {
        subInputDDR.push_back(createDeclareTensorOp(functionBuilder, VPURT::BufferSection::DDR, subInputShapes[index],
                                                    inputType, vpux::DimsOrder::NHWC,
                                                    INPUT_DDR_OFFSET + INPUT_DDR_STRIDE * index));
        subOutputDDR.push_back(createDeclareTensorOp(functionBuilder, VPURT::BufferSection::DDR, subOutputShapes[index],
                                                     outputType, vpux::DimsOrder::NHWC,
                                                     OUTPUT_DDR_OFFSET + OUTPUT_DDR_STRIDE * index));
    }

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
            static_cast<std::int32_t>(0xFFFFFF), vpux::VPU::ArchKind::KMB, output.shape[1], weightsType);

    const auto weightsTableDDRMemRef =
            getMemRefType(VPURT::BufferSection::Constant, weightsTableShape, int32, DimsOrder::NCHW)
                    .cast<vpux::NDTypeInterface>();
    const auto weightsTableValues =
            mlir::DenseElementsAttr::get(weightsTableDDRType, llvm::makeArrayRef<std::int32_t>(weightsTable));
    auto weightsTableDDR = functionBuilder.create<vpux::Const::DeclareOp>(
            builder.getUnknownLoc(), weightsTableDDRMemRef,
            vpux::Const::ContentAttr::get(weightsTableValues).reorder(vpux::DimsOrder::NCHW));

    // Define CMX buffer
    auto parentInputCMX = functionBuilder.create<VPURT::DeclareBufferOp>(
            builder.getUnknownLoc(), inputDistribute, VPURT::BufferSection::CMX_NN, sectionIndex, INPUT_CMX_OFFSET);
    auto parentOutputCMX = functionBuilder.create<VPURT::DeclareBufferOp>(
            builder.getUnknownLoc(), outputDistribute, VPURT::BufferSection::CMX_NN, sectionIndex, OUTPUT_CMX_OFFSET);
    auto parentWeightsCMX = functionBuilder.create<VPURT::DeclareBufferOp>(
            builder.getUnknownLoc(), weightsDistribute, VPURT::BufferSection::CMX_NN, sectionIndex, WEIGHTS_CMX_OFFSET);
    auto parentWeightsTableCMX = functionBuilder.create<VPURT::DeclareBufferOp>(
            builder.getUnknownLoc(), weightsTableDistribute, VPURT::BufferSection::CMX_NN, sectionIndex,
            WEIGHTSTABLE_CMX_OFFSET);

    llvm::SmallVector<VPURT::DeclareBufferOp> subInputCMX;
    llvm::SmallVector<VPURT::DeclareBufferOp> subOutputCMX;
    llvm::SmallVector<VPURT::DeclareBufferOp> subWeightsCMX;
    llvm::SmallVector<VPURT::DeclareBufferOp> subWeightsTableCMX;

    for (auto index = 0; index < numCluster; index++) {
        subInputCMX.push_back(createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN,
                                                    subInputShapes[index], inputType, vpux::DimsOrder::NHWC, index,
                                                    INPUT_CMX_OFFSET));
        subOutputCMX.push_back(createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN,
                                                     subOutputShapes[index], outputType, vpux::DimsOrder::NHWC, index,
                                                     OUTPUT_CMX_OFFSET));
        subWeightsCMX.push_back(createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsCMXShape,
                                                      weightsType, vpux::DimsOrder::NHWC, index, WEIGHTS_CMX_OFFSET));
        subWeightsTableCMX.push_back(createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN,
                                                           weightsTableShape, int32, DimsOrder::NCHW, index,
                                                           WEIGHTSTABLE_CMX_OFFSET));
    }

    int barrierNumber = 0;

    // Reorder input
    auto updateBarrier =
            functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);
    const auto inputMemPerm = vpux::getPermutationFromOrders(DimsOrder::NCHW, DimsOrder::NHWC, ctx);
    VPURT::wrapIntoTaskOp<VPUIP::PermuteUPAOp>(
            functionBuilder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.barrier()), builder.getUnknownLoc(),
            functionInput, parentInputDDR.getOperation()->getResult(0), inputMemPerm);
    VPURT::ConfigureBarrierOp waitInputReorderBarrier = updateBarrier;

    // Copy input from DDR to CMX
    updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);
    for (auto index = 0; index < numCluster; index++) {
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                functionBuilder, waitInputReorderBarrier.barrier(), updateBarrier.barrier(), builder.getUnknownLoc(),
                subInputDDR[index].getOperation()->getResult(0), subInputCMX[index].getOperation()->getResult(0));
    }
    VPURT::ConfigureBarrierOp waitInputToCMXBarrier = updateBarrier;

    // Move weights and weights table from DDR to CMX
    updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
            functionBuilder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.barrier()), builder.getUnknownLoc(),
            weightsDDR.getOperation()->getResult(0), parentWeightsCMX.getOperation()->getResult(0));
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
            functionBuilder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.barrier()), builder.getUnknownLoc(),
            weightsTableDDR.getOperation()->getResult(0), parentWeightsTableCMX.getOperation()->getResult(0));

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

        const auto start =
                getIntArrayAttr(ctx, std::vector<std::int64_t>{0, subInputShapes[index][HeightIndex] * index, 0});
        const auto end = getIntArrayAttr(
                ctx, std::vector<std::int64_t>{outputShape[3] - 1, subInputShapes[index][2] * (index + 1) - 1,
                                               outputShape[1] - 1});
        const auto pad = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                             paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);

        nceTask.addDPUTask(functionBuilder, start, end, pad, conv.cube_mode);
    }

    VPURT::ConfigureBarrierOp waitDPUTaskBarrier = updateBarrier;

    // copy output from CMX to DDR
    updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);
    for (auto index = 0; index < numCluster; index++) {
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, waitDPUTaskBarrier.barrier(), updateBarrier.barrier(),
                                              builder.getUnknownLoc(), subOutputCMX[index].getOperation()->getResult(0),
                                              subOutputDDR[index].getOperation()->getResult(0));
    }

    // reorder output
    VPURT::ConfigureBarrierOp waitCMXToDDRBarrier = updateBarrier;
    const auto outMemPerm = vpux::getPermutationFromOrders(DimsOrder::NHWC, DimsOrder::NCHW, ctx);
    VPURT::wrapIntoTaskOp<VPUIP::PermuteUPAOp>(functionBuilder, waitCMXToDDRBarrier.barrier(), mlir::ValueRange(),
                                               builder.getUnknownLoc(), parentOutputDDR.getOperation()->getResult(0),
                                               functionOutput, outMemPerm);

    functionBuilder.create<mlir::ReturnOp>(builder.getUnknownLoc(), functionOutput);

    module.dump();

    mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPU::createInitCompilerPass(VPU::ArchKind::KMB, VPU::CompilationMode::DefaultHW, None, log));
    if (conv.compress) {
        pm.addPass(VPUIP::createCompressWeightsPass(log));
    }

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    buildCNNOp(builder, function.getName(),
               {getTensorType(ShapeRef(inputShape), inputType, vpux::DimsOrder::NCHW, nullptr)},
               {getTensorType(ShapeRef(outputShape), outputType, vpux::DimsOrder::NCHW, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
