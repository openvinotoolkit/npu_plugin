//
// Copyright 2021 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <climits>
#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Support/DebugStringHelper.h>

#include "vpux/compiler/dialect/VPUIP/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace hwtest {

void buildMaxPool(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                  Logger& log, mlir::Type inputType, mlir::Type outputType) {
    auto* ctx = builder.getContext();

    auto input = testDesc.getInputLayer();
    auto pool_op = testDesc.getPoolLayer();
    auto output = testDesc.getOutputLayer();

    SmallVector<int64_t> in_shape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> out_shape(output.shape.begin(), output.shape.end());

    std::vector<int64_t> filter_size{pool_op.kernel_shape.at(0), pool_op.kernel_shape.at(1)};
    std::vector<int64_t> stride_vec(pool_op.stride.begin(), pool_op.stride.end());
    std::vector<int64_t> padding_vec = convertNBPadtoNCETaskPad(pool_op.pad);

    auto output_totalsize = totalTensorSize(out_shape, outputType);

    const auto OUTPUT_CMX_OFFSET = 0;
    const auto INPUT0_CMX_OFFSET = OUTPUT_CMX_OFFSET + output_totalsize;

    SmallVector<mlir::Type> inputTypes;
    const auto inputAffineMaps = DimsOrder::NHWC.toAffineMapsList(ctx, Shape(in_shape));
    inputTypes.push_back(
            getMemRefType(builder, VPUIP::MemoryLocation::ProgrammableInput, in_shape, inputType, inputAffineMaps));

    const auto outputAffineMaps = DimsOrder::NHWC.toAffineMapsList(ctx, Shape(out_shape));
    auto outputParamType =
            getMemRefType(builder, VPUIP::MemoryLocation::ProgrammableOutput, out_shape, outputType, outputAffineMaps);
    inputTypes.push_back(outputParamType);
    SmallVector<ArrayRef<mlir::AffineMap>> argsAffineMaps{inputAffineMaps, outputAffineMaps};

    const auto funcType = builder.getFunctionType(makeArrayRef(inputTypes), outputParamType);

    auto func = builder.create<mlir::FuncOp>(builder.getUnknownLoc(),
                                             llvm::formatv("maxpool_{0}_{1}", inputType, outputType).str(), funcType,
                                             builder.getStringAttr("private"));

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    // BUild VPUIP ops
    auto funcinput0 = func.getArgument(0);
    auto funcoutput = func.getArgument(1);

    // input - output cmx tensors
    auto input0cmx_type =
            getMemRefType(builder, VPUIP::MemoryLocation::VPU_CMX_NN, in_shape, inputType, inputAffineMaps);
    auto input0cmx = createDeclareTensorOp(funcbuilder, input0cmx_type, 0, INPUT0_CMX_OFFSET);

    auto outputcmx_type =
            getMemRefType(builder, VPUIP::MemoryLocation::VPU_CMX_NN, out_shape, outputType, outputAffineMaps);
    auto outputcmx = createDeclareTensorOp(funcbuilder, outputcmx_type, 0, OUTPUT_CMX_OFFSET);

    auto parent_input0cmx = createDeclareTensorOp(funcbuilder, input0cmx_type, 0, INPUT0_CMX_OFFSET);
    auto parent_outputcmx = createDeclareTensorOp(funcbuilder, outputcmx_type, 0, OUTPUT_CMX_OFFSET);

    // barrier config
    auto barrier0 = funcbuilder.create<VPUIP::ConfigureBarrierOp>(builder.getUnknownLoc(), 0);
    auto barrier1 = funcbuilder.create<VPUIP::ConfigureBarrierOp>(builder.getUnknownLoc(), 1);

    // DMA input-->cmx
    funcbuilder.create<VPUIP::NNDMAOp>(builder.getUnknownLoc(), funcinput0, input0cmx.getOperation()->getResult(0),
                                       mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()), false);

    // NCE Task
    auto filtersize = getIntArrayAttr(builder, filter_size);
    auto strides = getIntArrayAttr(builder, stride_vec);
    auto kernel_padding = getIntArrayAttr(builder, padding_vec);

    auto nceTask = funcbuilder.create<VPUIP::NCEClusterTaskOp>(
            builder.getUnknownLoc(), outputcmx_type, input0cmx.getOperation()->getResult(0), mlir::Value(),
            mlir::Value(), mlir::Value(),
            parent_input0cmx.getOperation()->getResult(0), parent_outputcmx.getOperation()->getResult(0),
            outputcmx.getOperation()->getResult(0), mlir::ValueRange(barrier0.barrier()),
            mlir::ValueRange(barrier1.barrier()), VPUIP::NCETaskType::MAXPOOL, filtersize, strides, kernel_padding,
            /*actChannelLength*/ nullptr, /*is_continued*/ nullptr);

    nceTask.addPPETask(funcbuilder);

    // Create DPU task for NCE task
    nceTask.variants().emplaceBlock();
    auto variantbuilder = mlir::OpBuilder::atBlockBegin(&nceTask.variants().front(), builder.getListener());

    std::vector<int32_t> start_vec{0, 0, 0};
    auto start = getIntArrayAttr(builder, start_vec);
    std::vector<int32_t> end_vec{static_cast<int32_t>(out_shape[3] - 1), static_cast<int32_t>(out_shape[2] - 1),
                                 static_cast<int32_t>(out_shape[1] - 1)};
    auto end = getIntArrayAttr(builder, end_vec);
    auto pad = VPUIP::PaddingAttr::get(getIntAttr(builder, padding_vec[PAD_NCETASK_LEFT]),
                                       getIntAttr(builder, padding_vec[PAD_NCETASK_RIGHT]),
                                       getIntAttr(builder, padding_vec[PAD_NCETASK_TOP]),
                                       getIntAttr(builder, padding_vec[PAD_NCETASK_BOTTOM]), ctx);

    variantbuilder.create<VPUIP::DPUTaskOp>(builder.getUnknownLoc(), start, end, pad, VPUIP::MPEMode::CUBOID_16x16);
    funcbuilder.create<VPUIP::NNDMAOp>(builder.getUnknownLoc(), outputcmx.getOperation()->getResult(0), funcoutput,
                                       mlir::ValueRange(barrier1.barrier()), mlir::ValueRange(), false);

    funcbuilder.create<mlir::ReturnOp>(builder.getUnknownLoc(), funcoutput);
    // set runtime resources
    mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(createSetCompileParamsPass(VPUIP::ArchKind::MTL, VPUIP::CompilationMode::ReferenceSW, None, log));

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");
    // IE.CNNNetwork
    buildCNNOp(builder, func.getName(), {getTensorType(in_shape, inputType, DimsOrder::NHWC)},
               {getTensorType(out_shape, outputType, DimsOrder::NHWC)});
}

}  // namespace hwtest
}  // namespace vpux
