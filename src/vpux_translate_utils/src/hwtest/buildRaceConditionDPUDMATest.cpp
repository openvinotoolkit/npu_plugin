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
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {
namespace hwtest {

//
//             [input]
//                |
//            (barrier)
//            |       |
//         (conv)    (DMAop)
//            |       |
//            (barrier)
//            |       |
//        ... (loop with conv ops, DMA ops and barriers)
//            |       |
//         (conv)    (DMAop)
//            |       |
//            (barrier)
//            |       |
//       [output0]  [output1]
//

void buildRaceConditionDPUDMATest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                                  mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type weightsType,
                                  mlir::Type outputType) {
    auto* ctx = builder.getContext();
    auto loc = builder.getUnknownLoc();
    const auto int32 = builder.getIntegerType(32, true);

    const auto input = testDesc.getInputLayer();
    const auto weights = testDesc.getWeightLayer();
    const auto conv = testDesc.getConvLayer();
    const auto output = testDesc.getOutputLayer();
    const auto iterationCount = testDesc.getIterationCount();

    const SmallVector<std::int64_t> inputShape{input.shape.begin(), input.shape.end()};
    const SmallVector<std::int64_t> outputShape{output.shape.begin(), output.shape.end()};
    const SmallVector<std::int64_t> weightsShape{weights.shape.begin(), weights.shape.end()};
    const SmallVector<std::int64_t> weightsTableShape{weightsShape[0], 1, 1, 4};

    VPUX_THROW_UNLESS(!inputShape.empty(), "buildRaceConditionDPUDMATest: Got empty inputShape");
    VPUX_THROW_UNLESS(!outputShape.empty(), "buildRaceConditionDPUDMATest: Got empty outputShape");
    VPUX_THROW_UNLESS(!weightsShape.empty(), "buildRaceConditionDPUDMATest: Got empty weightsShape");
    VPUX_THROW_UNLESS(!weightsTableShape.empty(), "buildRaceConditionDPUDMATest: Got empty weightsTableShape");

    const char* weightsFileName = "weights.dat";

    auto inputCMXShape = inputShape;

    auto weightsCMXShape = weightsShape;
    auto outputCMXShape = outputShape;

    const auto alignmentRequirement = 16;

    const auto weightsCMXSize = vpux::hwtest::totalTensorSize(weightsCMXShape, weightsType);
    const auto outputCMXSize = vpux::hwtest::totalTensorSize(outputCMXShape, outputType);
    const auto inputCMXSize = vpux::hwtest::totalTensorSize(inputCMXShape, inputType);

    const auto alignment =
            (alignmentRequirement * static_cast<vpux::Bit>(getElemTypeSize(inputType)).count()) / CHAR_BIT;
    const auto WEIGHTS_CMX_OFFSET = 0;
    VPUX_THROW_UNLESS(WEIGHTS_CMX_OFFSET % alignment == 0, "WEIGHTS_CMX_OFFSET must be multiple of {0}, got {1}",
                      alignment, WEIGHTS_CMX_OFFSET);

    const auto OUTPUT_CMX_OFFSET_0 = WEIGHTS_CMX_OFFSET + weightsCMXSize;
    VPUX_THROW_UNLESS(OUTPUT_CMX_OFFSET_0 % alignment == 0, "OUTPUT_CMX_OFFSET_0 must be multiple of {0}, got {1}",
                      alignment, OUTPUT_CMX_OFFSET_0);
    const auto OUTPUT_CMX_OFFSET_1 = OUTPUT_CMX_OFFSET_0 + outputCMXSize;
    VPUX_THROW_UNLESS(OUTPUT_CMX_OFFSET_1 % alignment == 0, "OUTPUT_CMX_OFFSET_1 must be multiple of {0}, got {1}",
                      alignment, OUTPUT_CMX_OFFSET_1);

    const auto INPUT_CMX_OFFSET = OUTPUT_CMX_OFFSET_1 + inputCMXSize;
    VPUX_THROW_UNLESS(INPUT_CMX_OFFSET % alignment == 0, "INPUT_CMX_OFFSET must be multiple of {0}, got {1}", alignment,
                      INPUT_CMX_OFFSET);

    const auto WEIGHTSTABLE_CMX_OFFSET = INPUT_CMX_OFFSET + inputCMXSize;
    VPUX_THROW_UNLESS(WEIGHTSTABLE_CMX_OFFSET % alignment == 0,
                      "WEIGHTSTABLE_CMX_OFFSET must be multiple of {0}, got {1}", alignment, WEIGHTSTABLE_CMX_OFFSET);

    const auto inputParamType =
            getMemRefType(VPURT::BufferSection::NetworkInput, inputShape, inputType, DimsOrder::NHWC);
    const auto outputParamType =
            getMemRefType(vpux::VPURT::BufferSection::NetworkOutput, outputShape, outputType, DimsOrder::NHWC);

    const auto funcType =
            builder.getFunctionType(SmallVector<mlir::Type>{inputParamType, outputParamType, inputParamType},
                                    SmallVector<mlir::Type>{outputParamType, inputParamType});

    auto function = builder.create<mlir::FuncOp>(
            loc, llvm::formatv("race_condition_dpu_dma_{0}_{1}_{2}", inputType, weightsType, outputType).str(),
            funcType, builder.getStringAttr("private"));

    auto functionBuilder = mlir::OpBuilder::atBlockBegin(function.addEntryBlock(), builder.getListener());

    auto functionInput = function.getArgument(0);
    auto functionOutput_0 = function.getArgument(1);
    auto functionOutput_1 = function.getArgument(2);

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
            getMemRefType(VPURT::BufferSection::Constant, weightsShape, weightsType, DimsOrder::NHWC);

    auto weightsStrides = vpux::getStrides(weightsDDRType);
    auto inputStrides = vpux::getStrides(functionInput.getType().cast<mlir::ShapedType>());

    auto weightsCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsShape, weightsType,
                                            DimsOrder::OYXI, weightsStrides, 0, WEIGHTS_CMX_OFFSET);
    auto inputCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, inputShape, inputType,
                                          DimsOrder::NHWC, inputStrides, 0, INPUT_CMX_OFFSET);

    auto weightsDDR = functionBuilder.create<vpux::Const::DeclareOp>(loc, weightsDDRType, weightsAttribute);

    auto outputCMX_0 = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, outputShape, outputType,
                                             DimsOrder::NHWC, 0, OUTPUT_CMX_OFFSET_0);
    auto outputCMX_1 = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, inputShape, inputType,
                                             DimsOrder::NHWC, 0, OUTPUT_CMX_OFFSET_1);

    auto& weightsOutputChannelsStrideInBits = weightsStrides[vpux::Dims4D::Filter::OC];

    const auto weightsTableDDRType = mlir::RankedTensorType::get(weightsTableShape, int32);
    const auto weightsTable = VPU::NCESparsity::getWeightsTable(
            inputType, outputType, static_cast<std::int32_t>(WEIGHTS_CMX_OFFSET),
            static_cast<std::int32_t>(weightsOutputChannelsStrideInBits.count() / CHAR_BIT),
            static_cast<std::int32_t>(0xFFFFFF), vpux::VPU::ArchKind::MTL, output.shape[1], weightsType);

    const auto weightsTableDDRMemRef =
            getMemRefType(VPURT::BufferSection::Constant, weightsTableShape, int32, DimsOrder::NHWC);
    const auto weightsTableValues =
            mlir::DenseElementsAttr::get(weightsTableDDRType, llvm::makeArrayRef<std::int32_t>(weightsTable));
    auto weightsTableDDR = functionBuilder.create<vpux::Const::DeclareOp>(
            loc, weightsTableDDRMemRef,
            vpux::Const::ContentAttr::get(weightsTableValues).reorder(vpux::DimsOrder::NHWC));

    auto weightsTableCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsTableShape,
                                                 int32, DimsOrder::NHWC, 0, WEIGHTSTABLE_CMX_OFFSET);

    auto updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, 0);
    VPURT::ConfigureBarrierOp waitBarrier;

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(),
                                          mlir::ValueRange(updateBarrier.barrier()), loc, functionInput,
                                          inputCMX.getOperation()->getResult(0));
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
            functionBuilder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.barrier()), loc,
            weightsDDR.getOperation()->getResult(0), weightsCMX.getOperation()->getResult(0));
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
            functionBuilder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.barrier()), loc,
            weightsTableDDR.getOperation()->getResult(0), weightsTableCMX.getOperation()->getResult(0));

    waitBarrier = updateBarrier;

    const auto strides = getIntArrayAttr(ctx, conv.stride);
    std::vector<std::int64_t> paddings = convertNBPadtoNCETaskPad(conv.pad);
    const auto kernelPaddings = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                                    paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);
    SmallVector<std::int64_t> kernel = {weightsShape[2], weightsShape[3]};
    const auto kernelSize = getIntArrayAttr(ctx, kernel);

    for (std::size_t i = 1; i < iterationCount; ++i) {
        updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, i);
        auto nceTask_0 = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
                functionBuilder, mlir::ValueRange(waitBarrier.barrier()), mlir::ValueRange(updateBarrier.barrier()),
                loc, inputCMX.buffer(), weightsCMX.buffer(), weightsTableCMX.buffer(), nullptr, inputCMX.buffer(),
                outputCMX_0.buffer(), outputCMX_0.buffer(), vpux::VPUIP::NCETaskType::CONV, kernelSize, strides,
                kernelPaddings, nullptr, nullptr);

        const auto start = getIntArrayAttr(ctx, std::vector<std::int64_t>{0, 0, 0});
        const auto end = getIntArrayAttr(
                ctx, std::vector<std::int64_t>{outputShape[3] - 1, outputShape[2] - 1, outputShape[1] - 1});
        const auto pad = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                             paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);
        nceTask_0.addDPUTask(functionBuilder, start, end, pad, conv.cube_mode);

        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                functionBuilder, mlir::ValueRange(waitBarrier.barrier()), mlir::ValueRange(updateBarrier.barrier()),
                loc, inputCMX.getOperation()->getResult(0), outputCMX_1.getOperation()->getResult(0), 0, false, false);
        waitBarrier = updateBarrier;
    }

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(waitBarrier.barrier()), mlir::ValueRange(),
                                          loc, outputCMX_0.getOperation()->getResult(0), functionOutput_0);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(waitBarrier.barrier()), mlir::ValueRange(),
                                          loc, outputCMX_1.getOperation()->getResult(0), functionOutput_1);

    functionBuilder.create<mlir::ReturnOp>(loc, mlir::ValueRange{functionOutput_0, functionOutput_1});

    module.dump();

    mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPU::createInitCompilerPass(VPU::ArchKind::MTL, VPU::CompilationMode::DefaultHW, None, log));
    if (conv.compress) {
        pm.addPass(VPUIP::createCompressWeightsPass(log));
    }

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    buildCNNOp(builder, function.getName(),
               {getTensorType(ShapeRef(inputShape), inputType, vpux::DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(outputShape), outputType, vpux::DimsOrder::NHWC, nullptr),
                getTensorType(ShapeRef(inputShape), inputType, vpux::DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
