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

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <llvm/Support/FormatVariadic.h>

//
// Generated
//

#include <vpux/compiler/dialect/VPU/generated/attributes/enums.hpp.inc>

namespace vpux {
namespace VPU {

//
// Run-time resources
//

StringLiteral getMemoryDerateAttrName();
StringLiteral getMemoryBandwidthAttrName();
StringLiteral getProcessorFrequencyAttrName();

uint32_t getMaxDPUClusterNum(mlir::Operation* op);

//
// ArchKind
//

void setArch(mlir::ModuleOp module, ArchKind kind, Optional<int> numOfDPUGroups = None);
ArchKind getArch(mlir::Operation* op);

//
// MemoryKind
//

MemoryKind getMemoryKind(mlir::RankedTensorType tensor);
MemoryKind getMemoryKind(mlir::MemRefType memref);
MemoryKind getMemoryKind(mlir::ShapedType type);

template <MemoryKind mem>
class MemoryResource final : public mlir::SideEffects::Resource::Base<MemoryResource<mem>> {
public:
    StringRef getName() final {
        return _name;
    }

private:
    friend typename MemoryResource::BaseT;

    MemoryResource() {
        _name = llvm::formatv("VPU.{0}", mem);
    }

private:
    std::string _name;
};

mlir::SideEffects::Resource* getMemoryResource(MemoryKind mem);
mlir::SideEffects::Resource* getMemoryResource(mlir::MemRefType memref);

//
// CompilationMode
//

void setCompilationMode(mlir::ModuleOp module, CompilationMode compilationMode);
CompilationMode getCompilationMode(mlir::Operation* op);

}  // namespace VPU
}  // namespace vpux
