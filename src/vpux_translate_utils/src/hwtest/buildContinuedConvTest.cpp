//
// Copyright 2021 Intel Corporation.
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

#include "vpux/compiler/dialect/VPUIP/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
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

    const auto int32 = builder.getIntegerType(32, true);

    const auto input = testDesc.getInputLayer();
    const auto weights = testDesc.getWeightLayer();
    const auto conv = testDesc.getConvLayer();
    const auto output = testDesc.getOutputLayer();

    const llvm::SmallVector<std::int64_t> inputShape(input.shape.begin(), input.shape.end());
    const llvm::SmallVector<std::int64_t> outputShape(output.shape.begin(), output.shape.end());
    const llvm::SmallVector<std::int64_t> weightsShape{weights.shape[0], weights.shape[1], weights.shape[2],
                                                       weights.shape[3]};

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

    const auto getMemRef = [&builder](const llvm::SmallVector<std::int64_t>& shape, mlir::Type type,
                                      vpux::VPUIP::MemoryLocation location) {
        const auto memSpaceAttr = vpux::VPUIP::MemoryLocationAttr::get(builder.getContext(), location);
        const auto affineMaps = vpux::DimsOrder::NHWC.toAffineMapsList(builder.getContext(), vpux::ShapeRef{shape});
        return mlir::MemRefType::get(llvm::makeArrayRef(shape), type, affineMaps, memSpaceAttr);
    };

    const auto outputParamType = getMemRef(outputShape, outputType, vpux::VPUIP::MemoryLocation::ProgrammableOutput);

    llvm::SmallVector<mlir::Type, 3> inputTypes;
    inputTypes.push_back(getMemRef(inputShape, inputType, vpux::VPUIP::MemoryLocation::ProgrammableInput));
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(llvm::makeArrayRef(inputTypes), outputParamType);

    auto function = builder.create<mlir::FuncOp>(
            builder.getUnknownLoc(),
            llvm::formatv("continued_conv_{0}_{1}_{2}", inputType, weightsType, outputType).str(), funcType,
            builder.getStringAttr("private"));

    auto functionBuilder = mlir::OpBuilder::atBlockBegin(function.addEntryBlock(), builder.getListener());

    const auto getCMXTensor = [&builder, &functionBuilder, getMemRef](const llvm::SmallVector<std::int64_t>& shape,
                                                                      mlir::Type type, std::size_t offset) {
        const auto CMXType = getMemRef(shape, type, vpux::VPUIP::MemoryLocation::VPU_CMX_NN);
        return functionBuilder.create<vpux::VPUIP::DeclareTensorOp>(builder.getUnknownLoc(), CMXType,
                                                                    vpux::VPUIP::MemoryLocation::VPU_CMX_NN, 0, offset);
    };

    const auto getMACAccTensor = [&builder, &functionBuilder, getMemRef](const llvm::SmallVector<std::int64_t>& shape,
                                                                         mlir::Type type, std::size_t offset) {
        const auto MACAccType = getMemRef(shape, type, vpux::VPUIP::MemoryLocation::MAC_Accumulators);
        return functionBuilder.create<vpux::VPUIP::DeclareTensorOp>(
                builder.getUnknownLoc(), MACAccType, vpux::VPUIP::MemoryLocation::MAC_Accumulators, 0, offset);
    };

    auto functionInput = function.getArgument(0);
    auto functionOutput = function.getArgument(1);

    // weights data
    const auto weightsValues = generateWeights(weightsShape, weightsType, builder.getContext(), weightsFileName);

    // Weights partial 0
    const auto weightsPartialParamType =
            getMemRef(weightsPartialShape, weightsType, vpux::VPUIP::MemoryLocation::VPU_CMX_NN);
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
            splitWeightsOverC(weightsValues, weightsShape, weightsType, builder.getContext(),
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
    const auto weightsTable0 = vpux::VPUIP::NCESparsity::getWeightsTable(
            inputType, outputType, static_cast<std::int32_t>(WEIGHTS_PARTIAL_0_CMX_OFFSET),
            static_cast<std::int32_t>(weightsPartialShape[1] * weightsPartialShape[2] * weightsPartialShape[3] *
                                      getElemTypeSize(weightsType).count() / 8),
            static_cast<std::int32_t>(ALL_BITS_ARE_SET), vpux::VPUIP::ArchKind::MTL, outputShape[1], weightsType);

    const auto weightsTableDDRMemRef = getMemRef(weightsTableShape, int32, vpux::VPUIP::MemoryLocation::GraphFile);
    const auto weightsTable0Values =
            mlir::DenseElementsAttr::get(weightsTableDDRType, llvm::makeArrayRef<std::int32_t>(weightsTable0));
    auto weightsTable0DDR = functionBuilder.create<vpux::Const::DeclareOp>(
            builder.getUnknownLoc(), weightsTableDDRMemRef,
            vpux::Const::ContentAttr::get(weightsTable0Values).reorder(vpux::DimsOrder::NHWC));
    auto weightsTable0CMX = getCMXTensor(weightsTableShape, int32, WEIGHTSTABLE_0_CMX_OFFSET);

    // weights table 1
    const auto weightsTable1 = vpux::VPUIP::NCESparsity::getWeightsTable(
            inputType, outputType, static_cast<std::int32_t>(WEIGHTS_PARTIAL_1_CMX_OFFSET),
            static_cast<std::int32_t>(weightsPartialShape[1] * weightsPartialShape[2] * weightsPartialShape[3] *
                                      getElemTypeSize(weightsType).count() / 8),
            static_cast<std::int32_t>(ALL_BITS_ARE_SET), vpux::VPUIP::ArchKind::MTL, outputShape[1], weightsType);

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
        auto barrier = functionBuilder.create<ConfigureBarrierOp>(builder.getUnknownLoc(), i);
        barriers.push_back(barrier.barrier());
    }

    // Input DMAs
    functionBuilder.create<NNDMAOp>(builder.getUnknownLoc(), functionInput, inputCMX.getOperation()->getResult(0),
                                    mlir::ValueRange(), barriers[0], false);
    functionBuilder.create<NNDMAOp>(builder.getUnknownLoc(), weightsPartial0DDR.getOperation()->getResult(0),
                                    weightsPartial0CMX.getOperation()->getResult(0), mlir::ValueRange(), barriers[0],
                                    false);
    functionBuilder.create<NNDMAOp>(builder.getUnknownLoc(), weightsPartial1DDR.getOperation()->getResult(0),
                                    weightsPartial1CMX.getOperation()->getResult(0), mlir::ValueRange(), barriers[0],
                                    false);
    functionBuilder.create<NNDMAOp>(builder.getUnknownLoc(), weightsTable0DDR.getOperation()->getResult(0),
                                    weightsTable0CMX.getOperation()->getResult(0), mlir::ValueRange(), barriers[0],
                                    false);
    functionBuilder.create<NNDMAOp>(builder.getUnknownLoc(), weightsTable1DDR.getOperation()->getResult(0),
                                    weightsTable1CMX.getOperation()->getResult(0), mlir::ValueRange(), barriers[0],
                                    false);

    // NCE params
    const auto strides = getIntArrayAttr(builder.getContext(), conv.stride);
    std::vector<std::int64_t> paddings = convertNBPadtoNCETaskPad(conv.pad);
    const auto kernelPaddings = getIntArrayAttr(builder.getContext(), paddings);
    llvm::SmallVector<std::int64_t> kernel = {weightsShape[2], weightsShape[3]};
    const auto kernelSize = getIntArrayAttr(builder.getContext(), kernel);
    const auto isContinued = mlir::UnitAttr::get(builder.getContext());

    // NCE Task 0
    auto nceTask_0 = functionBuilder.create<NCEClusterTaskOp>(
            builder.getUnknownLoc(), inputPartial0CMX.memory(), weightsPartial0CMX.memory(), weightsTable0CMX.memory(),
            /*activation_window=*/nullptr, inputPartial0CMX.memory(), output0CMX.memory(), output0CMX.memory(),
            VPUIP::NCETaskType::CONV, kernelSize, strides, kernelPaddings,
            /*activation_window_channel_length=*/nullptr, isContinued, /*odu_permutation=*/nullptr,
            /*sp_pattern*/ nullptr);

    // DPU task 0
    nceTask_0.waitBarriersMutable().append(barriers[0]);
    nceTask_0.updateBarriersMutable().append(barriers[1]);

    const auto start = getIntArrayAttr(builder.getContext(), std::vector<std::int64_t>{0, 0, 0});
    const auto end =
            getIntArrayAttr(builder.getContext(),
                            std::vector<std::int64_t>{outputShape[3] - 1, outputShape[2] - 1, outputShape[1] - 1});
    const auto pad = vpux::VPUIP::PaddingAttr::get(vpux::getIntAttr(builder, paddings[PAD_NCETASK_LEFT]),
                                                   vpux::getIntAttr(builder, paddings[PAD_NCETASK_RIGHT]),
                                                   vpux::getIntAttr(builder, paddings[PAD_NCETASK_TOP]),
                                                   vpux::getIntAttr(builder, paddings[PAD_NCETASK_BOTTOM]),
                                                   builder.getContext());

    nceTask_0.addDPUTask(functionBuilder, nullptr, start, end, pad, vpux::VPUIP::MPEMode::CUBOID_16x16);

    // NCE Task 1
    auto nceTask_1 = functionBuilder.create<NCEClusterTaskOp>(
            builder.getUnknownLoc(), inputPartial1CMX.memory(), weightsPartial1CMX.memory(), weightsTable1CMX.memory(),
            /*activation_window=*/nullptr, inputPartial1CMX.memory(), output1CMX.memory(), output1CMX.memory(),
            VPUIP::NCETaskType::CONV, kernelSize, strides, kernelPaddings,
            /*activation_window_channel_length=*/nullptr,
            /*is_continued*/ nullptr,
            /*odu_permutation=*/nullptr,
            /*sp_pattern*/ nullptr);

    // DPU task 1
    nceTask_1.waitBarriersMutable().append(barriers[1]);
    nceTask_1.updateBarriersMutable().append(barriers[2]);

    nceTask_1.addDPUTask(functionBuilder, nullptr, start, end, pad, vpux::VPUIP::MPEMode::CUBOID_16x16);

    functionBuilder.create<NNDMAOp>(builder.getUnknownLoc(), output1CMX.getOperation()->getResult(0), functionOutput,
                                    barriers[2], mlir::ValueRange(), false);

    functionBuilder.create<mlir::ReturnOp>(builder.getUnknownLoc(), functionOutput);

    mlir::PassManager pm(builder.getContext(), mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(vpux::VPUIP::createSetCompileParamsPass(vpux::VPUIP::ArchKind::MTL,
                                                       vpux::VPUIP::CompilationMode::ReferenceHW, None, log));

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    buildCNNOp(builder, function.getName(), {getTensorType(inputShape, inputType, vpux::DimsOrder::NHWC, nullptr)},
               {getTensorType(outputShape, outputType, vpux::DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
