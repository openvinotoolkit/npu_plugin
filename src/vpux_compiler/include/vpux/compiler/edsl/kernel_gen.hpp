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
namespace edsl {

static constexpr const uint8_t kDmaBeforeInner = 0x1;
static constexpr const uint8_t kDmaAfterInner = 0x2;
static constexpr const uint8_t kDmaBeforeMiddle = 0x4;
static constexpr const uint8_t kDmaAfterMiddle = 0x8;

struct MoviCompileParams {
    std::string cpu;
    std::string moviCompile;
    std::string mdkLinker;
    std::string mdkLibDir;
    std::vector<std::string> mdkLibs;
};

flatbuffers::Offset<MVCNN::BinaryData> generateKernelForSHAVE(mlir::FuncOp func, const MoviCompileParams& params,
                                                              flatbuffers::FlatBufferBuilder& fbb);

}  // namespace edsl
}  // namespace vpux
