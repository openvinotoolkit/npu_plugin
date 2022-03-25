//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/init.hpp"
#include "vpux/compiler/core/passes.hpp"
#include "vpux/compiler/utils/dot_printer.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>
#include <fstream>

namespace {

constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func @main(%arg0: memref<1x512xf32>, %arg1: memref<1x512xf32>) -> memref<1x512xf32> {
                %0 = memref.alloc() : memref<1x512xf32>
                %1 = IERT.SoftMax {axisInd = 1 : i32, test = 2 : i8} inputs(%arg0 : memref<1x512xf32>) outputs(%0 : memref<1x512xf32>) -> memref<1x512xf32>
                %2 = IERT.Copy inputs(%1 : memref<1x512xf32>) outputs(%arg1 : memref<1x512xf32>) -> memref<1x512xf32>
                memref.dealloc %0 : memref<1x512xf32>
                return %2 : memref<1x512xf32>
            }
        }
    )";

class CustomTestPass : public mlir::PassWrapper<CustomTestPass, vpux::ModulePass> {
public:
    ::llvm::StringRef getName() const override { return "TestPass"; }
    void safeRunOnModule() final {
    }
};

class CustomTestPass2 : public mlir::PassWrapper<CustomTestPass, vpux::ModulePass> {
public:
    ::llvm::StringRef getName() const override { return "TestPass2"; }
    void safeRunOnModule() final {
    }
};

static void CheckDotFile(const std::string fileName) {
    std::ifstream output_file(fileName);
    ASSERT_TRUE(output_file.good());
    std::string str;
    std::getline(output_file, str);
    ASSERT_TRUE(str.find("digraph") != std::string::npos);
}

} // namespace

TEST(MLIR_DotGraph, GenerateViaPass) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);
    mlir::MLIRContext ctx(registry);

    const std::string fileName = "output.dot";
    std::remove(fileName.c_str());

    auto module = mlir::parseSourceString(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    mlir::PassManager pm(&ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(vpux::createPrintDotPass(fileName));

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    CheckDotFile(fileName);
}

TEST(MLIR_DotGraph, GenerateViaEnvVar) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);
    mlir::MLIRContext ctx(registry);

    const std::string fileName = "output.dot";
    const std::string fileName2 = "output2.dot";
    std::remove(fileName.c_str());
    std::remove(fileName2.c_str());

    const std::string options = "output="+fileName+" pass=TestPass,output="+fileName2+" pass=TestPass2";

    auto module = mlir::parseSourceString(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    mlir::PassManager pm(&ctx, mlir::OpPassManager::Nesting::Implicit);
    vpux::addDotPrinter(pm, options);
    pm.addPass(std::make_unique<CustomTestPass>());
    pm.addPass(std::make_unique<CustomTestPass2>());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    CheckDotFile(fileName);
    CheckDotFile(fileName2);
}
