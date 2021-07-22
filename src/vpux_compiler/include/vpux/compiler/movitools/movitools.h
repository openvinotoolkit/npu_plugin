//
// Copyright 2021 Intel Corporation.
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
#include <vector>

#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinOps.h>

#include "vpux/compiler/dialect/VPUIP/schema.hpp"

namespace vpux {
namespace movitools {

struct MoviCompileParams {
    std::string cpu;
    std::string moviCompile;
    std::string mdkLinker;
    std::string mdkLibDir;
    std::vector<std::string> mdkLibs;
};

std::string getMoviToolsDir();

}  // namespace movitools
}  // namespace vpux
