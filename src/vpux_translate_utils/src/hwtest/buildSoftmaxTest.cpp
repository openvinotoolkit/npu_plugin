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

#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/utils/core/small_string.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;

namespace vpux {
namespace hwtest {

void buildSoftmax(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                  Logger& log, mlir::Type inputType, mlir::Type outputType) {

    auto* ctx = builder.getContext();

    //  Input/Output -----------------------------------------------------------

    auto input = testDesc.getInputLayer();
    auto output = testDesc.getOutputLayer();
    SmallVector<int64_t> in_shape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> out_shape(output.shape.begin(), output.shape.end());

    SmallVector<mlir::Type> inputTypes;
    auto inputParamType = getMemRefType(
        VPURT::BufferSection::NetworkInput, in_shape, inputType, DimsOrder::NHWC);
    inputTypes.push_back(inputParamType);
    auto outputParamType = getMemRefType(
        VPURT::BufferSection::NetworkOutput, out_shape, outputType, DimsOrder::NHWC);
    inputTypes.push_back(outputParamType);

    // Create built-in function ------------------------------------------------

    SmallString builtInFunctionName{"builtin_softmax"};
    SmallString kernelEntryName{"singleShaveSoftmax"};
    SmallString kernelSourceFileName{"single_shave_softmax.cpp"};
    auto builtInFunction = VPUIP::createBuiltInFunction(
        module, builtInFunctionName, inputTypes, kernelEntryName, kernelSourceFileName, log);

    //  Function ---------------------------------------------------------------

    const auto funcType = builder.getFunctionType(makeArrayRef(inputTypes), outputParamType);

    auto func = builder.create<mlir::FuncOp>(
        builder.getUnknownLoc(), llvm::formatv("softmax_{0}_{1}", inputType, outputType).str(), funcType,
        builder.getStringAttr("private"));

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    auto funcinput = func.getArgument(0);
    auto funcoutput = func.getArgument(1);

    //  Build main function: input/Output cmx ----------------------------------

    const auto outputcmx_offset = 0;
    const auto inputcmx_offset = outputcmx_offset + totalTensorSize(out_shape, outputType);

    auto inputcmx_type = getMemRefType(
        VPURT::BufferSection::CMX_NN, in_shape, inputType, DimsOrder::NHWC);
    vpux::VPURT::DeclareBufferOp inputcmx = createDeclareTensorOp(
        funcbuilder, VPURT::BufferSection::CMX_NN, in_shape, inputcmx_type, DimsOrder::NHWC, 0, inputcmx_offset);

    auto outputcmx_type = getMemRefType(
        VPURT::BufferSection::CMX_NN, out_shape, outputType, DimsOrder::NHWC);
    vpux::VPURT::DeclareBufferOp outputcmx = createDeclareTensorOp(
        funcbuilder, VPURT::BufferSection::CMX_NN, out_shape, outputcmx_type, DimsOrder::NHWC, 0, outputcmx_offset);

    //  Build main function: barriers ------------------------------------------

    auto barrier0 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), 0);
    auto barrier1 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), 1);

    // Spawn Task: Load --------------------------------------------------------

    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder,
                                                mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()),
                                                builder.getUnknownLoc(),
                                                funcinput, getTensorResult(inputcmx));

    // Spawn Task: Kernel ------------------------------------------------------

    auto kernelBuilder = [&](auto /*fn object*/ kernelTaskBody){
        auto taskOp = funcbuilder.create<vpux::VPURT::TaskOp>(
            funcbuilder.getUnknownLoc(),
            mlir::ValueRange(barrier0.barrier()),
            mlir::ValueRange(barrier1.barrier()));

        mlir::OpBuilder::InsertPoint lastInsertionPoint = funcbuilder.saveInsertionPoint();
        auto& block = taskOp.body().emplaceBlock();
        funcbuilder.setInsertionPointToStart(&block);

        kernelTaskBody();

        funcbuilder.restoreInsertionPoint(lastInsertionPoint);
    };

    kernelBuilder([&](){
        const int64_t tileIndex = 0;

        auto swKernelOp = funcbuilder.create<VPUIP::SwKernelOp>(funcbuilder.getUnknownLoc(),
                                                                         inputcmx.buffer(), outputcmx.buffer(),
                                                                         builtInFunction, getIntAttr(ctx, tileIndex));
        VPUIP::initSwKernel(swKernelOp, inputcmx.buffer(), outputcmx.buffer(), getIntAttr(ctx, tileIndex), log);
    });

    // Spawn Task: Store -------------------------------------------------------

    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder,
                                                mlir::ValueRange(barrier1.barrier()), mlir::ValueRange(),
                                                builder.getUnknownLoc(),
                                                getTensorResult(outputcmx), funcoutput);

    funcbuilder.create<mlir::ReturnOp>(builder.getUnknownLoc(), funcoutput);

    //  Pass Manager -----------------------------------------------------------

    mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPU::createInitCompilerPass(VPU::ArchKind::MTL, VPU::CompilationMode::ReferenceSW, None, log));
    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    //  CNN Operation ----------------------------------------------------------

    buildCNNOp(builder, func.getName(), {getTensorType(ShapeRef(in_shape), inputType, DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(out_shape), outputType, DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
