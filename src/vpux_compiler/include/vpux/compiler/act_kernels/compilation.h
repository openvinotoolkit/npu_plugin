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

#include "vpux/compiler/dialect/VPUIP/graph-schema/schema.hpp"
#include "vpux/utils/core/small_vector.hpp"

namespace vpux {

struct ActShaveCompileParams {
    std::vector<std::string> cpu;
};

struct KernelDataDesc {
    std::string name;
    //flatbuffers::Offset<MVCNN::KernelData> data;
    SmallVector<uint8_t> data;
    // unpadded size
    size_t size;
};

struct ActKernelDesc {
    KernelDataDesc text;
    KernelDataDesc data;
};

struct SerializedKernelDataDesc {
    std::string name;
    flatbuffers::Offset<MVCNN::KernelData> data;
    // unpadded size
    size_t size;
};

struct CompilationUnitDesc {
    mlir::StringRef name;
    mlir::StringRef entry;
};

ActKernelDesc compileKernelForACTShave(const CompilationUnitDesc& unitDesc,
                                       const ActShaveCompileParams& params);

const CompilationUnitDesc& managementKernelCompilationDesc();

ActKernelDesc compileManagementKernelForACTShave(const ActShaveCompileParams& params);

flatbuffers::Offset<MVCNN::KernelData> buildKernelData(flatbuffers::FlatBufferBuilder& fbb,
                                                       llvm::ArrayRef<uint8_t> content);

}  // namespace vpux
