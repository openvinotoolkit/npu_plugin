//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUIP_ATTR_INTERFACES
#define VPUX_COMPILER_DIALECT_VPUIP_ATTR_INTERFACES

include "mlir/IR/OpBase.td"

//
// ProfilingMetadataAttrInterface
//

def VPUIP_ProfilingMetadataAttrInterface : AttrInterface<"ProfilingMetadataAttr"> {
    let description = [{
        Interface to mark profiling metadata attributes.
    }];

    let cppNamespace = "vpux::VPUIP";
}

#endif
