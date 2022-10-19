//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/Dialect/Quant/QuantTypes.h>
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/ops/act_shave_op.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/small_string.hpp"

using namespace vpux;

namespace vpux {
namespace hwtest {

VPU::M2iColorFmt getM2iFmt(nb::M2iFmt fmt) {
    if (fmt == nb::M2iFmt::SP_NV12_8)
        return VPU::M2iColorFmt::SP_NV12_8;
    if (fmt == nb::M2iFmt::PL_YUV420_8)
        return VPU::M2iColorFmt::PL_YUV420_8;

    if (fmt == nb::M2iFmt::IL_RGB888)
        return VPU::M2iColorFmt::IL_RGB888;
    if (fmt == nb::M2iFmt::IL_BGR888)
        return VPU::M2iColorFmt::IL_BGR888;

    if (fmt == nb::M2iFmt::PL_RGB24)
        return VPU::M2iColorFmt::PL_RGB24;
    if (fmt == nb::M2iFmt::PL_BGR24)
        return VPU::M2iColorFmt::PL_BGR24;

    if (fmt == nb::M2iFmt::PL_FP16_RGB)
        return VPU::M2iColorFmt::PL_FP16_RGB;
    if (fmt == nb::M2iFmt::PL_FP16_BGR)
        return VPU::M2iColorFmt::PL_FP16_BGR;

    VPUX_THROW("getM2iFmt unsupported fmt");
}

void buildM2iTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                  Logger& log, mlir::Type inputType, mlir::Type outputType) {
    auto* ctx = builder.getContext();

    auto input = testDesc.getInputLayerList().front();
    auto output = testDesc.getOutputLayers().front();
    auto params = testDesc.getM2iLayer();

    // Drop quantization info
    if (inputType.dyn_cast<mlir::quant::QuantizedType>()) {
        inputType = mlir::quant::QuantizedType::castToStorageType(inputType);
    }
    if (outputType.dyn_cast<mlir::quant::QuantizedType>()) {
        outputType = mlir::quant::QuantizedType::castToStorageType(outputType);
    }

    SmallVector<int64_t> inShape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> outShape(output.shape.begin(), output.shape.end());

    VPUX_THROW_UNLESS(!inShape.empty(), "buildM2iTest: got empty inputShape");
    VPUX_THROW_UNLESS(!outShape.empty(), "buildM2iTest: got empty outputShape");

    if (params.doNorm) {
        VPUX_THROW_UNLESS(params.normCoefs.size() > 0, "buildM2iTest: norm coeffs missing");
    }

    auto outputTotalSize = totalTensorSize(outShape, outputType);
    auto inputTotalSize = totalTensorSize(inShape, inputType);

    SmallVector<mlir::Type> inputTypes;
    inputTypes.push_back(getMemRefType(VPURT::BufferSection::NetworkInput, inShape, inputType, DimsOrder::NCHW));

    auto outputParamType = getMemRefType(VPURT::BufferSection::NetworkOutput, outShape, outputType, DimsOrder::NCHW);
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(makeArrayRef(inputTypes), outputParamType);

    auto func = builder.create<mlir::FuncOp>(builder.getUnknownLoc(), llvm::formatv("m2i_test").str(), funcType,
                                             builder.getStringAttr("private"));

    auto funcBuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    auto normCoefs = params.doNorm ? getFPArrayAttr(funcBuilder, params.normCoefs) : nullptr;

    // Build VPUIP ops
    auto funcInput0 = func.getArgument(0);
    auto funcOutput = func.getArgument(1);

    size_t cmxOffset = 0;
    int barrierNumber = 0;

    // CMX buffs
    auto inCMXtype = getMemRefType(VPURT::BufferSection::CMX_NN, 0, inShape, inputType, DimsOrder::NCHW);
    auto inCMX = createDeclareTensorOp(funcBuilder, inCMXtype, VPURT::BufferSection::CMX_NN, 0, cmxOffset);
    cmxOffset += inputTotalSize;

    auto outCMXtype = getMemRefType(VPURT::BufferSection::CMX_NN, 0, outShape, outputType, DimsOrder::NCHW);
    auto outCMX = createDeclareTensorOp(funcBuilder, outCMXtype, VPURT::BufferSection::CMX_NN, 0, cmxOffset);
    cmxOffset += outputTotalSize;

    mlir::Value m2iInput = inCMX.getOperation()->getResult(0);
    mlir::Value m2iOutput = outCMX.getOperation()->getResult(0);

    // Barriers
    auto barrier1 = funcBuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);
    auto barrier2 = funcBuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);
    llvm::SmallVector<mlir::Value> barriers1;  // between 'DmaIn' and 'M2iTaskOp'
    llvm::SmallVector<mlir::Value> barriers2;  // between 'M2iTaskOp' and 'DmaOut'
    barriers1.emplace_back(barrier1.barrier());
    barriers2.emplace_back(barrier2.barrier());

    auto barr1 = llvm::ArrayRef<mlir::Value>(barriers1);
    auto barr2 = llvm::ArrayRef<mlir::Value>(barriers2);

    // DmaIn
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder,
                                          mlir::ValueRange(),       // waits
                                          mlir::ValueRange(barr1),  // updates
                                          builder.getUnknownLoc(),
                                          funcInput0,                           // src (DDR)
                                          inCMX.getOperation()->getResult(0));  // dst (CMX)
    // M2ITask
    VPURT::wrapIntoTaskOp<VPUIP::M2ITaskOp>(funcBuilder,              // 0) builder
                                            mlir::ValueRange(barr1),  // 1) wait-barrier
                                            mlir::ValueRange(barr2),  // 2) update-barrier
                                            builder.getUnknownLoc(),  // 3) loc
                                            // next: actual M2ITaskOp args (see 'builder')
                                            m2iInput, m2iOutput, params.doCsc, params.doNorm,
                                            getM2iFmt(params.iFmt),  // in fmt
                                            getM2iFmt(params.oFmt),  // out fmt
                                            normCoefs);              // norm coefs

    // DmaOut
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder,
                                          mlir::ValueRange(barr2),  // waits
                                          mlir::ValueRange(),       // updates
                                          builder.getUnknownLoc(),
                                          outCMX.getOperation()->getResult(0),  // src (CMX)
                                          funcOutput);                          // dst (DDR)

    funcBuilder.create<mlir::ReturnOp>(builder.getUnknownLoc(), funcOutput);

    // set runtime resources
    mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(
            VPU::createInitCompilerPass(testDesc.getArchitecture(), VPU::CompilationMode::DefaultHW, None, None, log));
    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    // IE.CNNNetwork
    buildCNNOp(builder, func.getName(), {getTensorType(ShapeRef(inShape), inputType, DimsOrder::NCHW, nullptr)},
               {getTensorType(ShapeRef(outShape), outputType, DimsOrder::NCHW, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
