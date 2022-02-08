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

#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

//
// BarrierResource
//

namespace vpux {
namespace VPURT {

struct BarrierResource final : public mlir::SideEffects::Resource::Base<BarrierResource> {
    StringRef getName() final {
        return "VPURT::Barrier";
    }
};

}  // namespace VPURT
}  // namespace vpux

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/VPURT/generated/types.hpp.inc>
#undef GET_TYPEDEF_CLASSES
