//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_ELF_PASSES
#define VPUX_COMPILER_DIALECT_ELF_PASSES

include "mlir/Pass/PassBase.td"

//
// Cleanup ELF Sections
//

def RemoveEmptyELFSections : PassBase<"remove-empty-elf-sections", "vpux::FunctionPass"> {
    let summary = "Cleans up empty sections in ELF";

    let description = [{
        The pass is part of the ELF pipeline.

        Current lowering logic may pre-create sections, without yet knowing if any op will eventually land in it. This
        causes some empty sections to be placed in the ELF. This pass should clean them up.
    }];

    let constructor = "vpux::ELF::createRemoveEmptyELFSectionsPass()";

    let dependentDialects = [
        "vpux::ELF::ELFDialect"
    ];
}

//
// UpdateELFSectionFlags
//

def UpdateELFSectionFlags : PassBase<"update-ELF-section-flags", "vpux::ModulePass"> {
    let summary = "Update ELF section flags";

    let description = [{
        The pass is part of the ELF pipeline.

        The pass will update the ELF section flags (secFlags attribute) for each relevant section based on which
        processors will access the given section at run time. This information is retrieved from the ops contained
        within the section via the ELF_BinaryOpInterface::getAccessingProcs interface method.

        A relevant section in the context of this pass is one that requires allocation and/or loading by the loader.
    }];

    let constructor = "vpux::ELF::createUpdateELFSectionFlagsPass()";

    let dependentDialects = [
        "vpux::Const::ConstDialect",
        "vpux::ELF::ELFDialect",
        "vpux::VPUIPRegMapped::VPUIPRegMappedDialect",
    ];
}

#endif
