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

#ifndef VPUX_COMPILER_DIALECT_ELF_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_ELF_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// TaskOpInterface
//

def ELF_OpInterface : OpInterface<"OpInterface"> {
    let description = "Interface for ELF Op";

    let cppNamespace = "vpux::ELF";

    let methods = [
        InterfaceMethod<
            "Serialize the Task to BLOB format",
            "void",
            //"serialize", (ins "std::vector<char>&":$buffer)
            //"serialize", (ins "elf::Writer&":$writer)
            "serialize", (ins "std::vector<elf::writer::Section*>":$ELFSection, "int":$ELFSectionIndex, "std::vector<elf::writer::Symbol*>":$ELFSymbol, "int":$ELFSymbolIndex)
        >,
    ];

    /*
    let extraTraitClassDeclaration = [{
        void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects) {
            vpux::VPUIPRegMapped::getTaskEffects($_op, effects);
        }

        mlir::Attribute getExecutor(uint32_t& numUnits) {
            return vpux::VPUIPRegMapped::getTaskOpExecutor($_op, numUnits);
        }
    }];
    */
}

#endif
