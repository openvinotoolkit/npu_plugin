//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/ELF/export.hpp"
#include "vpux/compiler/dialect/ELF/import.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/import.hpp"
#include "vpux/compiler/frontend/IE.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/hwtest/hwtest.hpp"

#include "vpux/utils/core/format.hpp"

// TODO: E66812, it should be sufficient to have warnings disabled for 3-rd parties
// in CMake but it does not work for early versions of MSVC 2019
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4146)
#endif
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <llvm/Support/TargetSelect.h>

#include <mlir/IR/Dialect.h>
#include <mlir/Support/MlirOptMain.h>
#include <mlir/Translation.h>

#include <llvm/Support/SourceMgr.h>

#include <cpp/ie_cnn_network.h>
#include <ie_core.hpp>

#include <fstream>
#include <iostream>

#include <cstdlib>

using namespace vpux;

namespace {

llvm::cl::opt<bool> vpuxProfiling("vpux-profiling", llvm::cl::desc("Add profilingOutput region to the imported IR"),
                                  llvm::cl::init(false));

llvm::cl::opt<bool> enableDummyOpReplacement{"dummy-op-replacement",
                                             llvm::cl::desc("Replace unsupported SW Kernel ops with Dummy ones"),
                                             llvm::cl::init(false)};

//
// import-IE
//

mlir::OwningOpRef<mlir::ModuleOp> importIE(llvm::SourceMgr& sourceMgr, mlir::MLIRContext* ctx) {
    mlir::DialectRegistry registry;
    registerDialects(registry);
    ctx->appendDialectRegistry(registry);

    if (sourceMgr.getNumBuffers() != 1) {
        printTo(llvm::errs(),
                "Invalid source file for IE IR, it has unsupported number of "
                "buffers {0}",
                sourceMgr.getNumBuffers());
        return nullptr;
    }

    const auto netFileName = sourceMgr.getMemoryBuffer(1)->getBufferIdentifier();
    if (netFileName.empty()) {
        printTo(llvm::errs(), "Invalid source file for IE IR, not a file");
        return nullptr;
    }

    InferenceEngine::Core ieCore;
    InferenceEngine::CNNNetwork cnnNet;

    try {
        cnnNet = ieCore.ReadNetwork(netFileName.str());
    } catch (const std::exception& ex) {
        printTo(llvm::errs(), "Failed to open IE IR {0} : {1}", netFileName, ex.what());
        return nullptr;
    }

    mlir::OwningOpRef<mlir::ModuleOp> module;

    try {
        mlir::DefaultTimingManager tm;
        auto rootTiming = tm.getRootScope();
        std::vector<vpux::PreProcessInfo> preProcInfo;
        // For VPUX37XX the ngraph transformations are different compared to the rest of the platforms
        // because scales do not need to be aligned. Running with VPU::ArchKind::UNKNOWN will align scales, which can
        // result in an accuracy drop for VPUX37XX.
        module = IE::importNetwork(ctx, cnnNet, preProcInfo, false, rootTiming, vpuxProfiling, enableDummyOpReplacement,
                                   VPU::ArchKind::UNKNOWN);
    } catch (const std::exception& ex) {
        printTo(llvm::errs(), "Failed to translate IE IR {0} to MLIR : {1}", netFileName, ex.what());
        return nullptr;
    }

    return module;
}

//
// import-VPUIP
//

mlir::OwningOpRef<mlir::ModuleOp> importVPUIP(llvm::SourceMgr& sourceMgr, mlir::MLIRContext* ctx) {
    mlir::DialectRegistry registry;
    registerDialects(registry);
    ctx->appendDialectRegistry(registry);

    if (sourceMgr.getNumBuffers() != 1) {
        printTo(llvm::errs(),
                "Invalid source file for blob, it has unsupported number of "
                "buffers {0}",
                sourceMgr.getNumBuffers());
        return nullptr;
    }

    const auto blobFileName = sourceMgr.getMemoryBuffer(1)->getBufferIdentifier();
    if (blobFileName.empty()) {
        printTo(llvm::errs(), "Invalid source file for blob, not a file");
        return nullptr;
    }

    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::ifstream blobStream(blobFileName.str(), std::ios::binary);
    auto blob = std::vector<char>(std::istreambuf_iterator<char>(blobStream), std::istreambuf_iterator<char>());

    try {
        module = VPUIP::importBlob(ctx, blob);
    } catch (const std::exception& ex) {
        printTo(llvm::errs(), "Failed to translate blob {0} to MLIR : {1}", blobFileName, ex.what());
        return nullptr;
    }

    return module;
}

//
// import-ELF
//

mlir::OwningOpRef<mlir::ModuleOp> importELF(llvm::SourceMgr& sourceMgr, mlir::MLIRContext* ctx) {
    mlir::DialectRegistry registry;
    registerDialects(registry);
    ctx->appendDialectRegistry(registry);

    if (sourceMgr.getNumBuffers() != 1) {
        printTo(llvm::errs(),
                "Invalid source file for elf, it has unsupported number of "
                "buffers {0}",
                sourceMgr.getNumBuffers());
        return nullptr;
    }

    const auto elfFileName = sourceMgr.getMemoryBuffer(1)->getBufferIdentifier();
    if (elfFileName.empty()) {
        printTo(llvm::errs(), "Invalid source file for elf, not a file");
        return nullptr;
    }

    mlir::OwningOpRef<mlir::ModuleOp> module;

    try {
        module = ELF::importELF(ctx, elfFileName.str());
    } catch (const std::exception& ex) {
        printTo(llvm::errs(), "Failed to translate elf {0} to MLIR : {1}", elfFileName, ex.what());
        return nullptr;
    }

    return module;
}

//
// export-VPUIP
//

mlir::LogicalResult exportELF(mlir::ModuleOp module, llvm::raw_ostream& output, StringRef /*outputFileName*/) {
    mlir::DefaultTimingManager tm;
    const auto buf = ELF::exportToELF(module);
    output.write(reinterpret_cast<const char*>(buf.data()), buf.size());
    return mlir::success();
}

//
// export-LLVMIR
//

int dumpLLVMIR(mlir::ModuleOp module, llvm::raw_ostream& output) {
    Logger log("dumpLLVMIR", vpux::LogLevel::Info);

    // Register the translation to LLVM IR with the MLIR context.
    mlir::registerLLVMDialectTranslation(*module->getContext());

    // Convert the module to LLVM IR in a new LLVM IR context.
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule) {
        log.error("Failed to emit LLVM IR\n");
        return -1;
    }

    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

    /// Optionally run an optimization pipeline over the llvm module.
    auto optPipeline = mlir::makeOptimizingTransformer(
            /*optLevel=*/0,  // enableOpt ? 3 : 0, // Note: seems small diffs on my tests
            /*sizeLevel=*/0,
            /*targetMachine=*/nullptr);
    if (auto err = optPipeline(llvmModule.get())) {
        log.error("Failed to optimize LLVM IR {0}\n", err);
        return -1;
    }

    output << "dumpLLVMIR() for output:\n" << *llvmModule << "\n";

    return 0;
}

mlir::LogicalResult exportLLVMIR(mlir::ModuleOp module, llvm::raw_ostream& output, StringRef /*outputFileName*/) {
    dumpLLVMIR(module, output);

    return mlir::success();
}

//
// export-EMU
//

}  // namespace

int main(int argc, char* argv[]) {
    try {
        mlir::TranslateToMLIRRegistration("import-IE", importIE);
        mlir::TranslateToMLIRRegistration("import-HWTEST", importHWTEST);
        mlir::TranslateToMLIRRegistration("import-VPUIP", importVPUIP);
        mlir::TranslateToMLIRRegistration("import-ELF", importELF);
        mlir::TranslateFromMLIRRegistration("export-ELF", exportELF, registerDialects);
        mlir::TranslateFromMLIRRegistration("export-LLVMIR", exportLLVMIR, registerDialects);

        return mlir::asMainReturnCode(mlir::mlirTranslateMain(argc, argv, "VPUX Translation Testing Tool"));
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
