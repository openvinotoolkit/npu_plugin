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

#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/Block.h>

namespace vpux {

struct GraphWriterParams final {
    std::string startAfter;
    std::string stopBefore;
    bool printConst = false;
    bool printDeclarations = false;
    bool htmlLike = true;
};

mlir::LogicalResult writeDotGraph(mlir::Block& block, StringRef fileName, const GraphWriterParams& params);

}  // namespace vpux
