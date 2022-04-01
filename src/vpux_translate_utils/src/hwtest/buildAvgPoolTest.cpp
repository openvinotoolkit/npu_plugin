//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include <functional>
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/ppe_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"

#include <climits>

namespace vpux {
namespace hwtest {

void buildAvgpool(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                  Logger& log, mlir::Type inputType, mlir::Type outputType) {
    auto* ctx = builder.getContext();
    auto loc = builder.getUnknownLoc();

    auto input = testDesc.getInputLayer();
    auto pool_op = testDesc.getPoolLayer();
    auto output = testDesc.getOutputLayer();

    SmallVector<int64_t> in_shape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> out_shape(output.shape.begin(), output.shape.end());

    VPUX_THROW_UNLESS(!in_shape.empty(), "buildAvgpool: Got empty inputShape");
    VPUX_THROW_UNLESS(!out_shape.empty(), "buildAvgpool: Got empty outputShape");

    std::vector<int64_t> filter_size{pool_op.kernel_shape.at(0), pool_op.kernel_shape.at(1)};
    std::vector<int64_t> stride_vec(pool_op.stride.begin(), pool_op.stride.end());
    std::vector<int64_t> padding_vec = convertNBPadtoNCETaskPad(pool_op.pad);

    auto output_totalsize = totalTensorSize(out_shape, outputType);

    SmallVector<int64_t> wt_data_shape{in_shape[1], 1, pool_op.kernel_shape.at(0), pool_op.kernel_shape.at(1)};

    auto scaleValue = 1 / double(pool_op.kernel_shape.at(0) * pool_op.kernel_shape.at(1));

    mlir::Type weightsType = inputType;

    int64_t clampLow = std::numeric_limits<int32_t>::min();
    int64_t clampHigh = std::numeric_limits<int32_t>::max();
    int64_t LreluMult = 1;
    int64_t LreluShift = 0;

    if (auto outElemQType = outputType.template dyn_cast<mlir::quant::QuantizedType>()) {
        const auto zps = extractScalesAndZeroPoints(outputType).second;

        clampLow = outElemQType.getStorageTypeMin() - zps.front();
        clampHigh = outElemQType.getStorageTypeMax() - zps.front();
    }

    if (auto qtype = inputType.dyn_cast<mlir::quant::QuantizedType>()) {
        auto inputStorageType = mlir::quant::QuantizedType::castToStorageType(qtype);
        int64_t zeroPoint = 0;

        if (inputStorageType.isUnsignedInteger(8)) {
            weightsType = mlir::quant::UniformQuantizedType::get(0, getUInt8Type(ctx), builder.getF32Type(), scaleValue,
                                                                 zeroPoint, 0, 1);
        } else if (inputStorageType.isSignedInteger(8)) {
            weightsType = mlir::quant::UniformQuantizedType::get(mlir::quant::QuantizationFlags::FlagValue::Signed,
                                                                 getSInt8Type(ctx), builder.getF32Type(), scaleValue,
                                                                 zeroPoint, 0, 1);
        } else {
            VPUX_THROW("Unsupported storage type for input quantized type. I8 or U8 is supported only");
        }
    }

    const auto OUTPUT_CMX_OFFSET = 0;
    const auto INPUT_CMX_OFFSET = OUTPUT_CMX_OFFSET + output_totalsize;

    SmallVector<mlir::Type> inputTypes;
    inputTypes.push_back(getMemRefType(VPURT::BufferSection::NetworkInput, in_shape, inputType, DimsOrder::NHWC));
    auto outputParamType = getMemRefType(VPURT::BufferSection::NetworkOutput, out_shape, outputType, DimsOrder::NHWC);
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(makeArrayRef(inputTypes), outputParamType);

    auto func = builder.create<mlir::FuncOp>(loc, printToString("avgPool_{0}_{1}", inputType, outputType), funcType,
                                             builder.getStringAttr("private"));

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    // Build VPUIP ops
    auto funcinput = func.getArgument(0);
    auto funcoutput = func.getArgument(1);

    // input - output cmx tensors

    auto inputcmx_type = getMemRefType(VPURT::BufferSection::CMX_NN, 0, in_shape, inputType, DimsOrder::NHWC);
    auto inputcmx =
            createDeclareTensorOp(funcbuilder, inputcmx_type, VPURT::BufferSection::CMX_NN, 0, INPUT_CMX_OFFSET);

    auto outputcmx_type = getMemRefType(VPURT::BufferSection::CMX_NN, 0, out_shape, outputType, DimsOrder::NHWC);
    auto outputcmx =
            createDeclareTensorOp(funcbuilder, outputcmx_type, VPURT::BufferSection::CMX_NN, 0, OUTPUT_CMX_OFFSET);

    auto parent_inputcmx =
            createDeclareTensorOp(funcbuilder, inputcmx_type, VPURT::BufferSection::CMX_NN, 0, INPUT_CMX_OFFSET);
    auto parent_outputcmx =
            createDeclareTensorOp(funcbuilder, outputcmx_type, VPURT::BufferSection::CMX_NN, 0, OUTPUT_CMX_OFFSET);

    // barrier config
    auto barrier0 = funcbuilder.create<VPURT::ConfigureBarrierOp>(loc, 0);
    auto barrier1 = funcbuilder.create<VPURT::ConfigureBarrierOp>(loc, 1);

    // DMAs
    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()),
                                                loc, funcinput, inputcmx.getOperation()->getResult(0));

    // NCE Task
    auto filtersize = getIntArrayAttr(funcbuilder, filter_size);
    auto strides = getIntArrayAttr(funcbuilder, stride_vec);
    auto kernel_padding = VPU::getPaddingAttr(ctx, padding_vec[PAD_NCETASK_LEFT], padding_vec[PAD_NCETASK_RIGHT],
                                              padding_vec[PAD_NCETASK_TOP], padding_vec[PAD_NCETASK_BOTTOM]);

    auto nceTask = vpux::VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
            funcbuilder, mlir::ValueRange(barrier0.barrier()), mlir::ValueRange(barrier1.barrier()), loc,
            outputcmx_type, inputcmx.getOperation()->getResult(0), mlir::Value(), mlir::Value(), mlir::Value(),
            parent_inputcmx.getOperation()->getResult(0), parent_outputcmx.getOperation()->getResult(0),
            outputcmx.getOperation()->getResult(0), VPUIP::NCETaskType::AVEPOOL, filtersize, strides, kernel_padding,
            /*actChannelLength*/ nullptr, /*is_continued*/ nullptr, /*sp_pattern*/ nullptr);

    // Since AvgPool operation doesn't have weights table it requires final quantization scaling
    // to be part of output tensor description. Scale vector will be placed in PPE block and
    // later used during NCE task serialization
    auto quantScale = VPU::calculateQuantScaleVectorForAvgPool(inputcmx_type, outputcmx_type, filter_size,
                                                               testDesc.getArchitecture());

    if (quantScale.hasValue()) {
        const auto scale = quantScale.getValue();

        const auto scaleApproximation = QuantizationApproximation(testDesc.getArchitecture(), scale);

        nceTask.addPPETask(funcbuilder, VPU::PPEMode::NOOP, clampLow, clampHigh, LreluMult, LreluShift,
                           scaleApproximation.mult(), scaleApproximation.shift());
    } else {
        nceTask.addPPETask(funcbuilder, VPU::PPEMode::NOOP, clampLow, clampHigh, LreluMult, LreluShift);
    }

    // Create DPU task for NCE task
    auto variantbuilder = mlir::OpBuilder::atBlockBegin(&nceTask.variants().front(), funcbuilder.getListener());

    std::vector<int32_t> start_vec{0, 0, 0};
    auto start = getIntArrayAttr(funcbuilder, start_vec);
    std::vector<int32_t> end_vec{static_cast<int32_t>(out_shape[3] - 1), static_cast<int32_t>(out_shape[2] - 1),
                                 static_cast<int32_t>(out_shape[1] - 1)};
    auto end = getIntArrayAttr(funcbuilder, end_vec);
    auto pad = VPU::getPaddingAttr(ctx, padding_vec[PAD_NCETASK_LEFT], padding_vec[PAD_NCETASK_RIGHT],
                                   padding_vec[PAD_NCETASK_TOP], padding_vec[PAD_NCETASK_BOTTOM]);

    // NB For pooling operations, NTHW_NTK=(16, 4) is the only mode supported by
    // the hardware; this corresponds to CUBOID_16x16.
    variantbuilder.create<VPUIP::DPUTaskOp>(loc, start, end, pad, VPU::MPEMode::CUBOID_16x16);

    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(barrier1.barrier()), mlir::ValueRange(),
                                                loc, outputcmx.getOperation()->getResult(0), funcoutput);

    funcbuilder.create<mlir::ReturnOp>(loc, funcoutput);

    // set runtime resources
    mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPU::createInitCompilerPass(testDesc.getArchitecture(), VPU::CompilationMode::DefaultHW, None, log));

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    // IE.CNNNetwork
    buildCNNOp(builder, func.getName(), {getTensorType(ShapeRef(in_shape), inputType, DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(out_shape), outputType, DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
