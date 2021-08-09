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

#include <unistd.h>

#include <llvm/Object/ELF.h>
#include <llvm/Support/MemoryBuffer.h>

#include "vpux/compiler/backend/VPUIP.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"

#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/BufferUtils.h"

namespace vpux {
namespace hwtest {

void buildActKernelTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                        Logger& log) {
    // Load and validate the ELF
    char cwd[1024];
    auto result = getcwd(cwd, 1024);
    std::cerr << "Act kernel test: cwd=" << cwd << " - " << result << "\n";
    auto kernelFilename = testDesc.getKernelFilename();
    std::string err;
    auto buffer = mlir::openInputFile(kernelFilename, &err);
    VPUX_THROW_UNLESS(buffer, "{0}", err);
    auto elf = llvm::object::ELF32LEFile::create(buffer->getBuffer());
    VPUX_THROW_UNLESS(elf, "Unable to interpret '{0}' as ELF32LE", kernelFilename);
    auto& header = elf->getHeader();
    VPUX_THROW_UNLESS(header.checkMagic() && header.getFileClass() == llvm::ELF::ELFCLASS32 && elf->isLE() &&
                              header.e_machine == llvm::ELF::EM_SPARC && header.e_type == llvm::ELF::ET_REL,
                      "'{0}' must be Sparc ELF32LE, relocatable", kernelFilename);

    // Turn the ELF into a global value
    auto elementsType = mlir::RankedTensorType::get(makeArrayRef(static_cast<int64_t>(buffer->getBufferSize())),
                                                    getUInt8Type(builder.getContext()));

    std::vector<std::uint8_t> bufData(buffer->getBufferStart(), buffer->getBufferEnd());
    auto elements = mlir::DenseElementsAttr::get(elementsType, makeArrayRef<std::uint8_t>(bufData));
    auto constantOp = builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), /* type, */ elements);
    mlir::GlobalCreator globalCreator(module);
    auto globalOp = globalCreator.getGlobalFor(constantOp);
    constantOp.erase();

    // Set up data types.
    SmallVector<int64_t, 4> ioShape{1, 1, 8, 8};
    constexpr int64_t inputCmxOffset = 0;
    constexpr int64_t outputCmxOffset = 2 * 8 * 8;

    const auto ioAffineMaps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(ioShape));

    // Build the function.
    // Note that we add a dummy output type because CNNNetwork requires it.
    auto inputMemrefType = mlir::MemRefType::get(
            makeArrayRef(ioShape), builder.getF16Type(), ioAffineMaps,
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::ProgrammableInput));
    auto outputMemrefType = mlir::MemRefType::get(
            makeArrayRef(ioShape), builder.getF16Type(), ioAffineMaps,
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::ProgrammableOutput));
    SmallVector<mlir::Type, 2> inputMemrefTypes{inputMemrefType, outputMemrefType};
    auto funcType = builder.getFunctionType(inputMemrefTypes, mlir::TypeRange{outputMemrefType});
    auto func = builder.create<mlir::FuncOp>(builder.getUnknownLoc(), "simple_activation_kernel", funcType,
                                             builder.getStringAttr("private"));
    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    // Set up the tensor types.
    SmallVector<mlir::Type, 2> inputTensorTypes{getTensorType(ioShape, builder.getF16Type(), DimsOrder::NHWC, nullptr),
                                                getTensorType(ioShape, builder.getF16Type(), DimsOrder::NHWC, nullptr)};
    SmallVector<mlir::Type, 2> outputTensorTypes{
            getTensorType(ioShape, builder.getF16Type(), DimsOrder::NHWC, nullptr)};

    // Declare input and output CMX tensors.
    auto cmxMemLoc = VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    auto cmxMemType = mlir::MemRefType::get(ioShape, builder.getF16Type(), ioAffineMaps, cmxMemLoc);
    auto inputCmx = funcbuilder.create<VPUIP::DeclareTensorOp>(builder.getUnknownLoc(), cmxMemType,
                                                               VPUIP::MemoryLocation::VPU_CMX_NN, 0, inputCmxOffset);
    auto outputCmx = funcbuilder.create<VPUIP::DeclareTensorOp>(builder.getUnknownLoc(), cmxMemType,
                                                                VPUIP::MemoryLocation::VPU_CMX_NN, 0, outputCmxOffset);

    // Set up barriers.
    auto barrier0 = funcbuilder.create<VPUIP::ConfigureBarrierOp>(builder.getUnknownLoc(), 0);
    auto barrier1 = funcbuilder.create<VPUIP::ConfigureBarrierOp>(builder.getUnknownLoc(), 1);

    // Load input into CMX.
    funcbuilder.create<VPUIP::NNDMAOp>(builder.getUnknownLoc(), func.getArgument(0),
                                       inputCmx.getOperation()->getResult(0), mlir::ValueRange{},
                                       mlir::ValueRange{barrier0.barrier()}, false);

    // Run the kernel.
    auto globalRef = mlir::SymbolRefAttr::get(builder.getContext(), globalOp.getName());
    SmallVector<mlir::Attribute, 2> kernelArgs{builder.getI64IntegerAttr(inputCmxOffset),
                                               builder.getI64IntegerAttr(outputCmxOffset)};
    funcbuilder.create<VPUIP::ActKernelOp>(builder.getUnknownLoc(), globalRef, globalRef,
                                           builder.getIntegerAttr(builder.getIntegerType(64, false), header.e_entry),
                                           builder.getArrayAttr(builder.getArrayAttr(kernelArgs)),
                                           mlir::ValueRange{barrier0.barrier()}, mlir::ValueRange{barrier1.barrier()});

    // Store output from CMX.
    funcbuilder.create<VPUIP::NNDMAOp>(builder.getUnknownLoc(), outputCmx.getOperation()->getResult(0),
                                       func.getArgument(1), mlir::ValueRange(barrier1.barrier()), mlir::ValueRange(),
                                       false);

    // Terminate the function.
    funcbuilder.create<mlir::ReturnOp>(builder.getUnknownLoc(), func.getArgument(1));

    // Set runtime resources
    mlir::PassManager pm(builder.getContext(), mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPUIP::createSetCompileParamsPass(VPUIP::ArchKind::MTL, VPUIP::CompilationMode(), None, log));

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    // IE.CNNNetwork
    auto mainFuncName = mlir::FlatSymbolRefAttr::get(builder.getContext(), func.getName());
    auto cnnOp = builder.create<IE::CNNNetworkOp>(builder.getUnknownLoc(), mainFuncName);
    cnnOp.inputsInfo().emplaceBlock();
    cnnOp.outputsInfo().emplaceBlock();

    auto inputsInfoBuilder = mlir::OpBuilder::atBlockBegin(&cnnOp.inputsInfo().front(), builder.getListener());
    for (size_t idx = 0; idx < inputTensorTypes.size(); ++idx) {
        auto inputName = llvm::formatv("input_{0}", idx);
        auto nameAttr = builder.getStringAttr(inputName);
        auto userTypeAttr = mlir::TypeAttr::get(inputTensorTypes[idx]);
        inputsInfoBuilder.create<IE::DataInfoOp>(builder.getUnknownLoc(), nameAttr, userTypeAttr);
    }

    auto outputsInfoBuilder = mlir::OpBuilder::atBlockBegin(&cnnOp.outputsInfo().front(), builder.getListener());
    for (size_t idx = 0; idx < outputTensorTypes.size(); ++idx) {
        auto outputName = llvm::formatv("output_{0}", idx);
        auto nameAttr = builder.getStringAttr(outputName);
        auto userTypeAttr = mlir::TypeAttr::get(outputTensorTypes[idx]);
        outputsInfoBuilder.create<IE::DataInfoOp>(builder.getUnknownLoc(), nameAttr, userTypeAttr);
    }
}

}  // namespace hwtest
}  // namespace vpux
