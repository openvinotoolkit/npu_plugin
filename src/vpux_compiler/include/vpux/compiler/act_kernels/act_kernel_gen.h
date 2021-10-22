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
#include "vpux/compiler/movitools/movitools.h"

namespace vpux {

struct KernelDataDesc {
    std::string name;
    flatbuffers::Offset<MVCNN::KernelData> data;
    size_t size;
};

struct ActKernelDesc {
    KernelDataDesc text;
    KernelDataDesc data;
};

const int MaxCompilationUnitSources = 4;
const int MaxExtraDefines = 2;
const int MaxExtraIncludePaths = 11;

struct CompilationUnitDesc {
    mlir::StringRef name;
    mlir::StringRef entry;
    mlir::StringRef codePath[MaxCompilationUnitSources] = {};
    mlir::StringRef defines[MaxExtraDefines] = {};
    mlir::StringRef includePaths[MaxExtraIncludePaths] = {};
};

ActKernelDesc compileKernelForACTShave(const CompilationUnitDesc& unitDesc,
                                       const movitools::MoviCompileParams& params,
                                       flatbuffers::FlatBufferBuilder& fbb);

ActKernelDesc compileManagementKernelForACTShave(const CompilationUnitDesc & unitDesc,
                                                 const movitools::MoviCompileParams& params,
                                                 flatbuffers::FlatBufferBuilder& fbb);

flatbuffers::Offset<MVCNN::KernelData> buildKernelData(flatbuffers::FlatBufferBuilder& fbb,
                                                       llvm::ArrayRef<uint8_t> content);


}  // namespace vpux
