//
// Copyright 2020 Intel Corporation.
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

#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"

#include "vpux/utils/core/format.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace vpux {
namespace VPUIP {

template <PhysicalMemory mem>
class MemoryResource final : public mlir::SideEffects::Resource::Base<MemoryResource<mem>> {
public:
    StringRef getName() final {
        return _name;
    }

private:
    friend typename MemoryResource::BaseT;

    MemoryResource() {
        _name = llvm::formatv("VPUIP::PhysicalMemory::{0}", mem);
    }

private:
    std::string _name;
};

mlir::SideEffects::Resource* getMemoryResource(PhysicalMemory mem);
mlir::FailureOr<mlir::SideEffects::Resource*> getMemoryResource(mlir::MemRefType memref);

struct BarrierResource final : public mlir::SideEffects::Resource::Base<BarrierResource> {
    StringRef getName() final {
        return "VPUIP::Barrier";
    }
};

}  // namespace VPUIP
}  // namespace vpux
