//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/ELF/export.hpp"
#include "vpux/compiler/dialect/EMU/graph-schema/export.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/export.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/import.hpp"
#include "vpux/compiler/frontend/IE.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/hwtest/hwtest.hpp"

#include "vpux/utils/core/format.hpp"

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

//
// import-IE
//

mlir::OwningModuleRef importIE(llvm::SourceMgr& sourceMgr, mlir::MLIRContext* ctx) {
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

    mlir::OwningModuleRef module;

    try {
        mlir::DefaultTimingManager tm;
        auto rootTiming = tm.getRootScope();
        std::vector<vpux::PreProcessInfo> preProcInfo;
        // for VPUX37XX and VPUX40XX the ngraph is different because scales are not needed to be align. 
        // Running with VPU::ArchKind::UNKNOWN will align scales, this can drop accuracy a bit for VPUX37XX and VPUX40XX.
        module = IE::importNetwork(ctx, cnnNet, preProcInfo, false, rootTiming, vpuxProfiling, VPU::ArchKind::UNKNOWN);
    } catch (const std::exception& ex) {
        printTo(llvm::errs(), "Failed to translate IE IR {0} to MLIR : {1}", netFileName, ex.what());
        return nullptr;
    }

    return module;
}

//
// import-VPUIP
//

mlir::OwningModuleRef importVPUIP(llvm::SourceMgr& sourceMgr, mlir::MLIRContext* ctx) {
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

    mlir::OwningModuleRef module;
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
// export-VPUIP
//

mlir::LogicalResult exportVPUIP(mlir::ModuleOp module, llvm::raw_ostream& output, StringRef /*outputFileName*/) {
    mlir::DefaultTimingManager tm;
    auto rootTiming = tm.getRootScope();
    const std::vector<std::shared_ptr<const ov::Node>> params;
    const std::vector<std::shared_ptr<const ov::Node>> results;
    std::vector<vpux::PreProcessInfo> preProcInfo;
    const auto buf = VPUIP::exportToBlob(module, rootTiming, preProcInfo, params, results);
    output.write(reinterpret_cast<const char*>(buf.data()), buf.size());
    return mlir::success();
}

mlir::LogicalResult exportELF(mlir::ModuleOp module, llvm::raw_ostream& output, StringRef /*outputFileName*/) {
    mlir::DefaultTimingManager tm;
    const auto buf = ELF::exportToELF(module);
    output.write(reinterpret_cast<const char*>(buf.data()), buf.size());
    return mlir::success();
}

//
// export-EMU
//

mlir::LogicalResult exportEMU(mlir::ModuleOp module, llvm::raw_ostream& output, StringRef /*outputFileName*/) {
    mlir::DefaultTimingManager tm;
    auto rootTiming = tm.getRootScope();
    const std::vector<std::shared_ptr<const ov::Node>> params;
    const std::vector<std::shared_ptr<const ov::Node>> results;
    std::vector<vpux::PreProcessInfo> preProcInfo;
    const auto buf = EMU::exportToBlob(module, rootTiming, preProcInfo, params, results);
    output.write(reinterpret_cast<const char*>(buf.data()), buf.size());
    return mlir::success();
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        mlir::TranslateToMLIRRegistration("import-IE", importIE);
        mlir::TranslateToMLIRRegistration("import-HWTEST", importHWTEST);
        mlir::TranslateToMLIRRegistration("import-VPUIP", importVPUIP);
        mlir::TranslateFromMLIRRegistration("export-VPUIP", exportVPUIP, registerDialects);
        mlir::TranslateFromMLIRRegistration("export-ELF", exportELF, registerDialects);
        mlir::TranslateFromMLIRRegistration("export-EMU", exportEMU, registerDialects);

        return mlir::asMainReturnCode(mlir::mlirTranslateMain(argc, argv, "VPUX Translation Testing Tool"));
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
