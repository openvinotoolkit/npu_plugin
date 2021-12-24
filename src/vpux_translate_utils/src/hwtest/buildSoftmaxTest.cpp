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

#include "vpux/compiler/backend/VPUIP.hpp"

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_string.hpp"

using namespace vpux;

mlir::SymbolRefAttr createBuiltInFunction(mlir::ModuleOp module, IERT::LayerOpInterface origOp,
                                          const IERT::KernelInfo& kernelInfo, const Logger& log) {
    auto* ctx = module.getContext();
    OpBuilderLogger builderLog(log);
    static constexpr StringLiteral vpuSwModuleName{"VPU.SW"};

    auto innerModule = module.lookupSymbol<mlir::ModuleOp>(vpuSwModuleName);
    // creating VPU.SW module if it is not yet created
    if (!innerModule) {
        auto mainModuleBuilder = mlir::OpBuilder::atBlockBegin(module.getBody(), &builderLog);
        innerModule = mainModuleBuilder.create<mlir::ModuleOp>(mlir::UnknownLoc::get(ctx), vpuSwModuleName);
    }

    SmallString builtInFunctionName{"builtin_"};
    auto nonNamespaceOpName = origOp->getName().getStringRef().slice(origOp->getName().getDialectNamespace().size() + 1,
                                                                     mlir::StringRef::npos);
    builtInFunctionName.append(nonNamespaceOpName);

    auto builtInFlatFunction = mlir::SymbolRefAttr::get(ctx, builtInFunctionName);
    auto builtInFunction = mlir::SymbolRefAttr::get(ctx, innerModule.getName().getValue(), {builtInFlatFunction});

    // check if this builtInFunction already created - consider names are unique - e.g. no overloads
    auto prebuiltFunction = innerModule.lookupSymbol<mlir::FuncOp>(builtInFunctionName);
    if (prebuiltFunction) {
        log.trace("Found builtin function: {0}", builtInFunctionName);
        return builtInFunction;
    }

    const auto convertToUnrankedType = [](mlir::Value operand) -> mlir::Type {
        auto type = operand.getType().dyn_cast_or_null<mlir::MemRefType>();
        VPUX_THROW_UNLESS(type != nullptr, "Only MemRef type is supported");

        return mlir::UnrankedMemRefType::get(type.getElementType(), type.getMemorySpace());
    };

    auto& args = kernelInfo.args;
    auto opInputs = origOp.getInputs();
    auto opResults = origOp->getResults();

    SmallVector<mlir::Type> inputTypes;
    std::transform(opInputs.begin(), opInputs.end(), std::back_inserter(inputTypes), convertToUnrankedType);
    std::transform(opResults.begin(), opResults.end(), std::back_inserter(inputTypes), convertToUnrankedType);
    std::transform(args.begin(), args.end(), std::back_inserter(inputTypes), [](mlir::Attribute arg) {
        return arg.getType();
    });

    const auto funcType = mlir::FunctionType::get(ctx, inputTypes, {});

    auto innerModuleBuilder = mlir::OpBuilder::atBlockBegin(innerModule.getBody(), &builderLog);
    auto buildInOp = innerModuleBuilder.create<mlir::FuncOp>(mlir::UnknownLoc::get(ctx), builtInFunctionName, funcType);

    // modifying attributes
    buildInOp.sym_visibilityAttr(mlir::StringAttr::get(ctx, "private"));

    buildInOp->setAttr("VPU.kernel_entry", mlir::StringAttr::get(ctx, kernelInfo.entryName));
    buildInOp->setAttr("VPU.kernel_code", mlir::StringAttr::get(ctx, kernelInfo.sourceFileName));

    log.trace("Added new builtin function: {0}", builtInFunctionName);
    return builtInFunction;
}

 void initSwKernel(vpux::VPUIP::SwKernelOp swKernelOp, mlir::ValueRange inputs, mlir::ValueRange outputBuffs,
                  ArrayRef<mlir::Attribute> args, const Logger& log) {
    OpBuilderLogger builderLog(log);
    auto* ctx = swKernelOp.getContext();
    auto& bodyRegion = swKernelOp.body();
    auto& swKernelBlock = bodyRegion.emplaceBlock();

    // embedding block args
    auto addBlockArgs = [&swKernelBlock](auto&& cnt) {
        for (auto&& arg : cnt) {
            swKernelBlock.addArgument(arg.getType());
        }
    };

    addBlockArgs(inputs);
    addBlockArgs(outputBuffs);

    auto swKernelBlockBuilder = mlir::OpBuilder::atBlockBegin(&swKernelBlock, &builderLog);

    // embedding args of IERT operation as constants
    SmallVector<mlir::arith::ConstantOp> constantArgs;
    for (auto&& arg : args) {
        constantArgs.push_back(swKernelBlockBuilder.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(ctx), arg));
    }

    // pack input/outputs and constants into single call to sw_kernel_run
    SmallVector<mlir::Value> operands;
    auto fetchOperands = [&operands](auto&& cnt) {
        for (auto&& arg : cnt) {
            operands.push_back(arg);
        }
    };

    auto blockArgs = swKernelBlock.getArguments();
    fetchOperands(blockArgs);
    fetchOperands(constantArgs);

    swKernelBlockBuilder.create<VPUIP::SwKernelRun>(mlir::UnknownLoc::get(ctx), mlir::ValueRange(operands));
}

#include <climits>
#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Support/DebugStringHelper.h>

#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"

#include "vpux/compiler/dialect/VPUIP/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace hwtest {

void buildSoftmax(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                  Logger& log, mlir::Type inputType, mlir::Type outputType) {

    auto* ctx = builder.getContext();

    //  Nested VPU.SW module ---------------------------------------------------

    // static constexpr StringLiteral vpuSwModuleName{"VPU.SW"};
    // auto vpuSwModule = mlir::ModuleOp::create(module.getLoc(), vpuSwModuleName);
    // auto vpuSwBuilder = mlir::OpBuilder(vpuSwModule.getBodyRegion());

    // auto vpuSwInput = testDesc.getInputLayer();
    // auto vpuSwOutput = testDesc.getOutputLayer();
    // SmallVector<int64_t> vpuSw_in_shape(vpuSwInput.shape.begin(), vpuSwInput.shape.end());
    // SmallVector<int64_t> vpuSw_out_shape(vpuSwOutput.shape.begin(), vpuSwOutput.shape.end());

    // SmallVector<mlir::Type> vpuSwInputTypes;
    // auto vpuSwInputParamType = getMemRefType(
    //     vpuSwBuilder, VPUIP::MemoryLocation::ProgrammableInput, vpuSw_in_shape, inputType, DimsOrder::NHWC); 
    // vpuSwInputTypes.push_back(vpuSwInputParamType);
    // auto vpuSwOutputParamType = getMemRefType(
    //     vpuSwBuilder, VPUIP::MemoryLocation::ProgrammableOutput, vpuSw_out_shape, outputType, DimsOrder::NHWC);
    // vpuSwInputTypes.push_back(vpuSwOutputParamType);

    // const auto vpuSwFuncType = vpuSwBuilder.getFunctionType(makeArrayRef(vpuSwInputTypes), vpuSwOutputParamType);

    // auto vpuSwFunc = vpuSwBuilder.create<mlir::FuncOp>(
    //     vpuSwBuilder.getUnknownLoc(), llvm::formatv("builtin_softmax_{0}_{1}", inputType, outputType).str(), vpuSwFuncType,
    //     vpuSwBuilder.getStringAttr("private"));

    // // auto vpuSwFuncbuilder = mlir::OpBuilder::atBlockBegin(vpuSwFunc.addEntryBlock(), vpuSwBuilder.getListener());
    // // auto vpuSwFuncinput = vpuSwFunc.getArgument(0);
    // // auto vpuSwFuncoutput = vpuSwFunc.getArgument(1);
    // // vpuSwFuncbuilder.create<mlir::ReturnOp>(vpuSwBuilder.getUnknownLoc(), vpuSwFuncoutput);

    //  Input/Output -----------------------------------------------------------

    auto input = testDesc.getInputLayer();
    auto output = testDesc.getOutputLayer();
    SmallVector<int64_t> in_shape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> out_shape(output.shape.begin(), output.shape.end());

    SmallVector<mlir::Type> inputTypes;
    auto inputParamType = getMemRefType(
        builder, VPUIP::MemoryLocation::ProgrammableInput, in_shape, inputType, DimsOrder::NHWC); 
    inputTypes.push_back(inputParamType);
    auto outputParamType = getMemRefType(
        builder, VPUIP::MemoryLocation::ProgrammableOutput, out_shape, outputType, DimsOrder::NHWC);
    inputTypes.push_back(outputParamType);

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
        builder, VPUIP::MemoryLocation::VPU_CMX_NN, in_shape, inputType, DimsOrder::NHWC);
    auto inputcmx = createDeclareTensorOp(funcbuilder, inputcmx_type, 0, inputcmx_offset);

    auto outputcmx_type = getMemRefType(
        builder, VPUIP::MemoryLocation::VPU_CMX_NN, out_shape, outputType, DimsOrder::NHWC);
    auto outputcmx = createDeclareTensorOp(funcbuilder, outputcmx_type, 0, outputcmx_offset);

    //  Build main function: barriers ------------------------------------------

    auto barrier0 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), 0);
    auto barrier1 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), 1);

    // SoftMaxOp ---------------------------------------------------------------

    // auto softmaxOp = funcbuilder.create<IERT::SoftMaxOp>(
    //     funcbuilder.getUnknownLoc(), inputcmx, outputcmx, 0);

    // IERT::KernelInfo kernelInfo = VPUIP::SwKernelOp::getKernelInfo(softmaxOp);
    // auto builtInFunction = createBuiltInFunction(module, softmaxOp, kernelInfo, log);

    auto swKernelOp = funcbuilder.create<VPUIP::SwKernelOp>(builder.getUnknownLoc(), inputcmx, outputcmx,
                                                            builtInFunction, getIntAttr(ctx, 0));

    // initSwKernel(softmaxOp, inputcmx, outputcmx, mlir::ValueRange(), log);

    // Load/Kernel/Store Tasks -------------------------------------------------

    vpux::VPURT::WrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()),
                                                builder.getUnknownLoc(), funcinput,
                                                inputcmx.getOperation()->getResult(0), false);

    // auto softMaxTask = vpux::VPURT::WrapIntoTaskOp<VPUIP::SwKernelOp>(
    //     funcbuilder,
    //     mlir::ValueRange(barrier0.barrier()),
    //     mlir::ValueRange(barrier1.barrier()),
    //     builder.getUnknownLoc(),
    //     inputcmx,
    //     outputcmx,
    //     0
    // );

    vpux::VPURT::WrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(barrier1.barrier()), mlir::ValueRange(),
                                                builder.getUnknownLoc(), outputcmx.getOperation()->getResult(0),
                                                funcoutput, true);

    funcbuilder.create<mlir::ReturnOp>(builder.getUnknownLoc(), funcoutput);

    //  Pass Manager -----------------------------------------------------------

    mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(createSetCompileParamsPass(VPUIP::ArchKind::MTL, VPUIP::CompilationMode::ReferenceSW, None, log));
    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    //  CNN Operation ----------------------------------------------------------

    buildCNNOp(builder, func.getName(), {getTensorType(ShapeRef(in_shape), inputType, DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(out_shape), outputType, DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
