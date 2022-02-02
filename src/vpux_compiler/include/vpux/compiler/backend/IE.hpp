//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include <string>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <ngraph/function.hpp>
#include "vpux/compiler/dialect/IE/ops.hpp"

namespace vpux {
namespace IE {

std::shared_ptr<ngraph::Function> exportToNgraph(IE::CNNNetworkOp, mlir::FuncOp netFunc);
mlir::LogicalResult exportToOpenVINO(mlir::ModuleOp module, llvm::raw_ostream&, const llvm::StringRef filePath);

}  // namespace IE
}  // namespace vpux
