//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace hwtest {

void buildEltwiseAdd(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                     Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType) {
    auto* ctx = builder.getContext();
    auto arch = testDesc.getArchitecture();

    auto input = testDesc.getInputLayerList().front();
    auto weight = testDesc.getWeightLayers().front();
    auto output = testDesc.getOutputLayers().front();

    SmallVector<int64_t> in_shape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> weights_shape(weight.shape.begin(), weight.shape.end());
    SmallVector<int64_t> out_shape(output.shape.begin(), output.shape.end());

    VPUX_THROW_UNLESS(in_shape.size() >= 4, "buildEltwiseAdd: Input rank is less than 4");
    VPUX_THROW_UNLESS(out_shape.size() >= 4, "buildEltwiseAdd: Output rank is less than 4");
    VPUX_THROW_UNLESS(weights_shape.size() >= 4, "buildEltwiseAdd: Weights rank is less than 4");

    auto output_totalsize = totalTensorSize(out_shape, outputType);
    auto input_totalsize = totalTensorSize(in_shape, inputType);

    const auto OUTPUT_CMX_OFFSET = 0;
    const auto INPUT0_CMX_OFFSET = OUTPUT_CMX_OFFSET + output_totalsize;
    const auto INPUT1_CMX_OFFSET = INPUT0_CMX_OFFSET + input_totalsize;

    VPUX_THROW_UNLESS((inputType == weightsType), "Eltwise expects inputs of same type");

    SmallVector<mlir::Type> inputTypes;
    inputTypes.push_back(getMemRefType(VPURT::BufferSection::NetworkInput, in_shape, inputType, DimsOrder::NHWC));
    inputTypes.push_back(
            getMemRefType(VPURT::BufferSection::NetworkInput, weights_shape, weightsType, DimsOrder::NHWC));

    auto outputParamType = getMemRefType(VPURT::BufferSection::NetworkOutput, out_shape, outputType, DimsOrder::NHWC);
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(makeArrayRef(inputTypes), outputParamType);

    auto func = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(), printToString("eltwise_{0}_{1}_{2}", inputType, weightsType, outputType), funcType,
            builder.getStringAttr("private"));

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    // Build VPUIP ops
    auto funcinput = func.getArgument(0);
    auto funcweights = func.getArgument(1);
    auto funcoutput = func.getArgument(2);

    // input - output cmx tensors
    auto inputcmx_type = getMemRefType(VPURT::BufferSection::CMX_NN, 0, in_shape, inputType, DimsOrder::NHWC);
    auto inputcmx =
            createDeclareTensorOp(funcbuilder, inputcmx_type, VPURT::BufferSection::CMX_NN, 0, INPUT0_CMX_OFFSET);

    auto weightscmx_type = getMemRefType(VPURT::BufferSection::CMX_NN, 0, weights_shape, weightsType, DimsOrder::NHWC);
    auto weightscmx =
            createDeclareTensorOp(funcbuilder, weightscmx_type, VPURT::BufferSection::CMX_NN, 0, INPUT1_CMX_OFFSET);

    auto outputcmx_type = getMemRefType(VPURT::BufferSection::CMX_NN, 0, out_shape, outputType, DimsOrder::NHWC);
    auto outputcmx =
            createDeclareTensorOp(funcbuilder, outputcmx_type, VPURT::BufferSection::CMX_NN, 0, OUTPUT_CMX_OFFSET);

    auto parent_inputcmx =
            createDeclareTensorOp(funcbuilder, inputcmx_type, VPURT::BufferSection::CMX_NN, 0, INPUT0_CMX_OFFSET);
    auto parent_outputcmx =
            createDeclareTensorOp(funcbuilder, outputcmx_type, VPURT::BufferSection::CMX_NN, 0, OUTPUT_CMX_OFFSET);

    // barrier config
    auto barrier0 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), 0);
    auto barrier1 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), 1);

    // DMAs
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()),
                                          builder.getUnknownLoc(), funcinput, inputcmx.getOperation()->getResult(0));
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()),
                                          builder.getUnknownLoc(), funcweights,
                                          weightscmx.getOperation()->getResult(0));
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(barrier1.barrier()), mlir::ValueRange(),
                                          builder.getUnknownLoc(), outputcmx.getOperation()->getResult(0), funcoutput);

    // NCE Task
    mlir::IntegerAttr actChannelLength = builder.getI32IntegerAttr(0);
    auto nceTask = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
            funcbuilder, mlir::ValueRange(barrier0.barrier()), mlir::ValueRange(barrier1.barrier()),
            builder.getUnknownLoc(), outputcmx_type, inputcmx.getOperation()->getResult(0),
            weightscmx.getOperation()->getResult(0), mlir::Value(), /*instruction_table_list=*/nullptr,
            /*activation_window=*/nullptr, parent_inputcmx.getOperation()->getResult(0),
            parent_outputcmx.getOperation()->getResult(0), outputcmx.getOperation()->getResult(0),
            VPUIP::NCETaskType::ELTWISE, mlir::ArrayAttr(), mlir::ArrayAttr(), VPU::PaddingAttr(), actChannelLength,
            /*is_continued*/ nullptr, /*sp_pattern*/ nullptr);

    int64_t clampLow = std::numeric_limits<int32_t>::min();
    int64_t clampHigh = std::numeric_limits<int32_t>::max();
    int64_t bypassMult = 1;
    int64_t bypassShift = 0;

    if (auto outElemQType = outputType.template dyn_cast<mlir::quant::QuantizedType>()) {
        const auto zps = extractScalesAndZeroPoints(outputType).second;

        clampLow = outElemQType.getStorageTypeMin() - zps.front();
        clampHigh = outElemQType.getStorageTypeMax() - zps.front();
    }

    // Since Eltwise operation doesn't have weights table it requires final quantization scaling
    // to be part of output tensor description. Scale vector will be placed in PPE block and
    // later used during NCE task serialization
    auto eltwiseQuantScale =
            VPU::calculateQuantScaleVectorForEltwise(inputcmx_type, weightscmx_type, outputcmx_type, arch, false);
    // Scale approximation is required for quantized inputs.
    if (inputcmx_type.getElementType().isa<mlir::FloatType>()) {
        // It is intentional to apply int32 limits for floating point clamping.
        // See E#50875 for details.
        nceTask.addPPETask(funcbuilder, VPU::PPEMode::ADD, clampLow, clampHigh, bypassMult, bypassShift, bypassMult,
                           bypassShift, bypassShift, eltwiseQuantScale);
    } else {
        const auto scaleApproximation = QuantizationApproximation(arch, eltwiseQuantScale);
        nceTask.addPPETask(funcbuilder, VPU::PPEMode::ADD, clampLow, clampHigh, bypassMult, bypassShift,
                           scaleApproximation.mult(), scaleApproximation.shift());
    }

    // Create DPU task for NCE task
    auto variantbuilder = mlir::OpBuilder::atBlockBegin(&nceTask.variants().front(), builder.getListener());

    std::vector<int32_t> start_vec{0, 0, 0};
    auto start = getIntArrayAttr(builder, start_vec);
    const auto outEnd =
            getIntArrayAttr(ctx, std::vector<std::int64_t>{out_shape[3] - 1, out_shape[2] - 1, out_shape[1] - 1});
    const auto inEnd =
            getIntArrayAttr(ctx, std::vector<std::int64_t>{in_shape[3] - 1, in_shape[2] - 1, in_shape[1] - 1});
    auto pad = VPU::getPaddingAttr(ctx, 0, 0, 0, 0);

    // NB For eltwise operations, NTHW_NTK=(8, 8) is the only mode supported by
    // the hardware; this corresponds to CUBOID_8x16.
    nceTask.addDPUTask(variantbuilder, start, outEnd, start, inEnd, pad, VPU::MPEMode::CUBOID_8x16);

    funcbuilder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), funcoutput);

    // set runtime resources
    mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPU::createInitCompilerPass(arch, VPU::CompilationMode::DefaultHW, 1, None, None, log));

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    // IE.CNNNetwork
    buildCNNOp(builder, func.getName(),
               {getTensorType(ShapeRef(in_shape), inputType, DimsOrder::NHWC, nullptr),
                getTensorType(ShapeRef(weights_shape), weightsType, DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(in_shape), outputType, DimsOrder::NHWC, nullptr)});
}
}  // namespace hwtest
}  // namespace vpux
