//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"

#include "vpux/utils/core/format.hpp"

#include <mlir/IR/StandardTypes.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace vpux {
namespace VPUIP {

template <MemoryLocation location>
class MemoryResource final
        : public mlir::SideEffects::Resource::Base<MemoryResource<location>> {
public:
    StringRef getName() final {
        return _name;
    }

private:
    friend typename MemoryResource::BaseT;

    MemoryResource() {
        _name = llvm::formatv("VPUIP::Memory::{0}", location);
    }

private:
    std::string _name;
};

mlir::SideEffects::Resource* getMemoryResource(MemoryLocation location);
mlir::SideEffects::Resource* getMemoryResource(mlir::MemRefType memref);

struct BarrierResource final
        : public mlir::SideEffects::Resource::Base<BarrierResource> {
    StringRef getName() final {
        return "VPUIP::Barrier";
    }
};

}  // namespace VPUIP
}  // namespace vpux
