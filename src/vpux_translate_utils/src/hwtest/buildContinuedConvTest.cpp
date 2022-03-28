//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"

#define ALL_BITS_ARE_SET 0xFFFFFF

namespace vpux {
namespace hwtest {

//
//       [input]
//          |
//       (conv_0) --- (conv_1)
//                       |
//                    [output]
//

void buildContinuedConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                        Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType) {
    using namespace VPUIP;

    auto* ctx = builder.getContext();
    const auto int32 = builder.getIntegerType(32, true);

    const auto input = testDesc.getInputLayer();
    const auto weights = testDesc.getWeightLayer();
    const auto conv = testDesc.getConvLayer();
    const auto output = testDesc.getOutputLayer();

    const llvm::SmallVector<std::int64_t> inputShape(input.shape.begin(), input.shape.end());
    const llvm::SmallVector<std::int64_t> outputShape(output.shape.begin(), output.shape.end());
    const llvm::SmallVector<std::int64_t> weightsShape{weights.shape[0], weights.shape[1], weights.shape[2],
                                                       weights.shape[3]};

    VPUX_THROW_UNLESS(inputShape.size() >= 4, "buildContinuedConv: Got inputShape with rank less than 4");
    VPUX_THROW_UNLESS(outputShape.size() >= 4, "buildContinuedConv: Got outputShape with rank less than 4");
    VPUX_THROW_UNLESS(weightsShape.size() >= 4, "buildContinuedConv: Got weightsShape with rank less than 4");

    const auto streamsOverC = 2;
    const llvm::SmallVector<std::int64_t> inputPartialShape(
            {inputShape[0], inputShape[1] / streamsOverC, inputShape[2], inputShape[3]});
    const llvm::SmallVector<std::int64_t> weightsPartialShape(
            {weightsShape[0], weightsShape[1] / streamsOverC, weightsShape[2], weightsShape[3]});
    const llvm::SmallVector<std::int64_t> weightsTableShape{weightsPartialShape[0], 1, 1, 4};

    const char* weightsFileName = "weights.dat";

    const auto OUTPUT_CMX_OFFSET = 0;
    const auto OUTPUT_CONV_0_CMX_OFFSET = OUTPUT_CMX_OFFSET + totalTensorSize(outputShape, outputType);
    const auto OUTPUT_CONV_1_CMX_OFFSET = OUTPUT_CONV_0_CMX_OFFSET + totalTensorSize(outputShape, outputType);
    const auto INPUT_CMX_OFFSET = OUTPUT_CONV_1_CMX_OFFSET + totalTensorSize(outputShape, outputType);
    const auto WEIGHTSTABLE_0_CMX_OFFSET = INPUT_CMX_OFFSET + totalTensorSize(inputShape, inputType);
    const auto WEIGHTSTABLE_1_CMX_OFFSET = WEIGHTSTABLE_0_CMX_OFFSET + 4 * weightsTableShape[0] * weightsTableShape[3];
    const auto WEIGHTS_PARTIAL_0_CMX_OFFSET =
            WEIGHTSTABLE_1_CMX_OFFSET + 4 * weightsTableShape[0] * weightsTableShape[3];
    const auto WEIGHTS_PARTIAL_1_CMX_OFFSET =
            WEIGHTS_PARTIAL_0_CMX_OFFSET + totalTensorSize(weightsPartialShape, weightsType);
    const auto INPUT_CONV_0_CMX_OFFSET = INPUT_CMX_OFFSET;
    const auto INPUT_CONV_1_CMX_OFFSET = INPUT_CONV_0_CMX_OFFSET + totalTensorSize(inputShape, inputType) / 2;

    const auto getMemRef = [&builder](ArrayRef<std::int64_t> shape, mlir::Type elemType, VPU::MemoryKind memKind) {
        return vpux::getMemRefType(ShapeRef(shape), elemType, DimsOrder::NHWC, memKind);
    };

    const auto outputParamType = getMemRef(outputShape, outputType, VPU::MemoryKind::DDR);

    llvm::SmallVector<mlir::Type, 3> inputTypes;
    inputTypes.push_back(getMemRef(inputShape, inputType, VPU::MemoryKind::DDR));
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(llvm::makeArrayRef(inputTypes), outputParamType);

    auto function = builder.create<mlir::FuncOp>(
            builder.getUnknownLoc(),
            llvm::formatv("continued_conv_{0}_{1}_{2}", inputType, weightsType, outputType).str(), funcType,
            builder.getStringAttr("private"));

    auto functionBuilder = mlir::OpBuilder::atBlockBegin(function.addEntryBlock(), builder.getListener());

    const auto getCMXTensor = [&builder, &functionBuilder, getMemRef](const llvm::SmallVector<std::int64_t>& shape,
                                                                      mlir::Type type, std::size_t offset) {
        const auto CMXType = getMemRef(shape, type, vpux::VPU::MemoryKind::CMX_NN);
        return functionBuilder.create<vpux::VPURT::DeclareBufferOp>(builder.getUnknownLoc(), CMXType,
                                                                    VPURT::BufferSection::CMX_NN, 0, offset);
    };

    const auto getMACAccTensor = [&builder, &functionBuilder, getMemRef](const llvm::SmallVector<std::int64_t>& shape,
                                                                         mlir::Type type, std::size_t offset) {
        const auto MACAccType = getMemRef(shape, type, VPU::MemoryKind::Register);
        return functionBuilder.create<vpux::VPURT::DeclareBufferOp>(builder.getUnknownLoc(), MACAccType,
                                                                    VPURT::BufferSection::MAC_Accumulators, 0, offset);
    };

    auto functionInput = function.getArgument(0);
    auto functionOutput = function.getArgument(1);

    // weights data
    const auto weightsValues = generateWeights(weightsShape, weightsType, ctx, weightsFileName);

    // Weights partial 0
    const auto weightsPartialParamType = getMemRef(weightsPartialShape, weightsType, vpux::VPU::MemoryKind::CMX_NN);
    auto weightsPartial0Values = splitWeightsOverC(weightsValues, weightsShape, weightsType, builder.getContext(),
                                                   /*startC*/ 0, /*endC*/ weightsPartialShape[1]);
    auto weightsPartial0Attribute = Const::ContentAttr::get(weightsPartial0Values);
    if (auto qty = weightsType.dyn_cast<mlir::quant::QuantizedType>()) {
        weightsPartial0Attribute = weightsPartial0Attribute.quantCast(qty);
    }
    auto weightsPartial0DDR = functionBuilder.create<Const::DeclareOp>(
            builder.getUnknownLoc(), weightsPartialParamType, weightsPartial0Attribute.reorder(DimsOrder::NHWC));

    // Weights partial 1
    auto weightsPartial1Values =
            splitWeightsOverC(weightsValues, weightsShape, weightsType, ctx,
                              /*startC*/ weightsPartialShape[1], /*endC*/ 2 * weightsPartialShape[1]);
    auto weightsPartial1Attribute = Const::ContentAttr::get(weightsPartial1Values);
    if (auto qty = weightsType.dyn_cast<mlir::quant::QuantizedType>()) {
        weightsPartial1Attribute = weightsPartial1Attribute.quantCast(qty);
    }
    auto weightsPartial1DDR = functionBuilder.create<Const::DeclareOp>(
            builder.getUnknownLoc(), weightsPartialParamType, weightsPartial1Attribute.reorder(DimsOrder::NHWC));

    auto inputCMX = getCMXTensor(inputShape, inputType, INPUT_CMX_OFFSET);

    // Tensors - NCE_0
    auto inputPartial0CMX = getCMXTensor(inputPartialShape, inputType, INPUT_CONV_0_CMX_OFFSET);
    auto weightsPartial0CMX = getCMXTensor(weightsPartialShape, weightsType, WEIGHTS_PARTIAL_0_CMX_OFFSET);
    auto output0CMX = getMACAccTensor(outputShape, outputType, OUTPUT_CONV_0_CMX_OFFSET);

    // Tensors - NCE_1
    auto inputPartial1CMX = getCMXTensor(inputPartialShape, inputType, INPUT_CONV_1_CMX_OFFSET);
    auto weightsPartial1CMX = getCMXTensor(weightsPartialShape, weightsType, WEIGHTS_PARTIAL_1_CMX_OFFSET);
    auto output1CMX = getCMXTensor(outputShape, outputType, OUTPUT_CONV_1_CMX_OFFSET);

    // weights table 0
    const auto weightsTableDDRType = mlir::RankedTensorType::get(weightsTableShape, int32);
    const auto weightsTable0 = VPU::NCESparsity::getWeightsTable(
            inputType, outputType, static_cast<std::int32_t>(WEIGHTS_PARTIAL_0_CMX_OFFSET),
            static_cast<std::int32_t>(weightsPartialShape[1] * weightsPartialShape[2] * weightsPartialShape[3] *
                                      getElemTypeSize(weightsType).count() / 8),
            static_cast<std::int32_t>(ALL_BITS_ARE_SET), testDesc.getArchitecture(), outputShape[1], weightsType);

    const auto weightsTableDDRMemRef = getMemRef(weightsTableShape, int32, VPU::MemoryKind::DDR);
    const auto weightsTable0Values =
            mlir::DenseElementsAttr::get(weightsTableDDRType, llvm::makeArrayRef<std::int32_t>(weightsTable0));
    auto weightsTable0DDR = functionBuilder.create<vpux::Const::DeclareOp>(
            builder.getUnknownLoc(), weightsTableDDRMemRef,
            vpux::Const::ContentAttr::get(weightsTable0Values).reorder(vpux::DimsOrder::NHWC));
    auto weightsTable0CMX = getCMXTensor(weightsTableShape, int32, WEIGHTSTABLE_0_CMX_OFFSET);

    // weights table 1
    const auto weightsTable1 = VPU::NCESparsity::getWeightsTable(
            inputType, outputType, static_cast<std::int32_t>(WEIGHTS_PARTIAL_1_CMX_OFFSET),
            static_cast<std::int32_t>(weightsPartialShape[1] * weightsPartialShape[2] * weightsPartialShape[3] *
                                      getElemTypeSize(weightsType).count() / 8),
            static_cast<std::int32_t>(ALL_BITS_ARE_SET), testDesc.getArchitecture(), outputShape[1], weightsType);

    const auto weightsTable1Values =
            mlir::DenseElementsAttr::get(weightsTableDDRType, llvm::makeArrayRef<std::int32_t>(weightsTable1));
    auto weightsTable1DDR = functionBuilder.create<vpux::Const::DeclareOp>(
            builder.getUnknownLoc(), weightsTableDDRMemRef,
            vpux::Const::ContentAttr::get(weightsTable1Values).reorder(vpux::DimsOrder::NHWC));
    auto weightsTable1CMX = getCMXTensor(weightsTableShape, int32, WEIGHTSTABLE_1_CMX_OFFSET);

    // Barriers
    std::vector<mlir::Value> barriers;
    auto num_barriers = 3;
    for (auto i = 0; i < num_barriers; ++i) {
        auto barrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), i);
        barriers.push_back(barrier.barrier());
    }

    // Input DMAs
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(), barriers[0], builder.getUnknownLoc(),
                                          functionInput, inputCMX.getOperation()->getResult(0));
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(), barriers[0], builder.getUnknownLoc(),
                                          weightsPartial0DDR.getOperation()->getResult(0),
                                          weightsPartial0CMX.getOperation()->getResult(0));
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(), barriers[0], builder.getUnknownLoc(),
                                          weightsPartial1DDR.getOperation()->getResult(0),
                                          weightsPartial1CMX.getOperation()->getResult(0));
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(), barriers[0], builder.getUnknownLoc(),
                                          weightsTable0DDR.getOperation()->getResult(0),
                                          weightsTable0CMX.getOperation()->getResult(0));
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(), barriers[0], builder.getUnknownLoc(),
                                          weightsTable1DDR.getOperation()->getResult(0),
                                          weightsTable1CMX.getOperation()->getResult(0));

    // NCE params
    const auto strides = getIntArrayAttr(ctx, conv.stride);
    std::vector<std::int64_t> paddings = convertNBPadtoNCETaskPad(conv.pad);
    const auto kernelPaddings = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                                    paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);

    llvm::SmallVector<std::int64_t> kernel = {weightsShape[2], weightsShape[3]};
    const auto kernelSize = getIntArrayAttr(ctx, kernel);
    const auto isContinued = mlir::UnitAttr::get(ctx);

    // NCE Task 0
    auto nceTask_0 = VPURT::wrapIntoTaskOp<NCEClusterTaskOp>(
            functionBuilder, barriers[0], barriers[1], builder.getUnknownLoc(), inputPartial0CMX.buffer(),
            weightsPartial0CMX.buffer(), weightsTable0CMX.buffer(),
            /*activation_window=*/nullptr, inputPartial0CMX.buffer(), output0CMX.buffer(), output0CMX.buffer(),
            VPUIP::NCETaskType::CONV, kernelSize, strides, kernelPaddings,
            /*activation_window_channel_length=*/nullptr, isContinued, /*sp_pattern*/ nullptr);

    const auto start = getIntArrayAttr(ctx, std::vector<std::int64_t>{0, 0, 0});
    const auto end =
            getIntArrayAttr(ctx, std::vector<std::int64_t>{outputShape[3] - 1, outputShape[2] - 1, outputShape[1] - 1});
    const auto pad = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                         paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);

    nceTask_0.addDPUTask(functionBuilder, start, end, pad, VPU::MPEMode::CUBOID_16x16);

    // NCE Task 1
    auto nceTask_1 = VPURT::wrapIntoTaskOp<NCEClusterTaskOp>(
            functionBuilder, barriers[1], barriers[2], builder.getUnknownLoc(), inputPartial1CMX.buffer(),
            weightsPartial1CMX.buffer(), weightsTable1CMX.buffer(),
            /*activation_window=*/nullptr, inputPartial1CMX.buffer(), output1CMX.buffer(), output1CMX.buffer(),
            VPUIP::NCETaskType::CONV, kernelSize, strides, kernelPaddings,
            /*activation_window_channel_length=*/nullptr,
            /*is_continued*/ nullptr, /*sp_pattern*/ nullptr);

    nceTask_1.addDPUTask(functionBuilder, start, end, pad, VPU::MPEMode::CUBOID_16x16);

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, barriers[2], mlir::ValueRange(), builder.getUnknownLoc(),
                                          output1CMX.getOperation()->getResult(0), functionOutput);

    functionBuilder.create<mlir::ReturnOp>(builder.getUnknownLoc(), functionOutput);

    mlir::PassManager pm(builder.getContext(), mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPU::createInitCompilerPass(testDesc.getArchitecture(), VPU::CompilationMode::DefaultHW, None, log));

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    buildCNNOp(builder, function.getName(),
               {getTensorType(ShapeRef(inputShape), inputType, vpux::DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(outputShape), outputType, vpux::DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
