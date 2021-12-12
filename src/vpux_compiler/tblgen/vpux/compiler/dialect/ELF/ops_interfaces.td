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
// ELFSectionInterface
//

def ELF_SectionInterface : OpInterface<"ElfSectionInterface"> {
    let description = "Interface for ELF section";

    let cppNamespace = "vpux::ELF";

    let methods = [
        InterfaceMethod<
            "Serialize the Section into an elf entity",
            "void",
            "serialize", (ins "elf::Writer&":$writer,
                            "vpux::OpOrderedMap<elf::writer::Section*>&":$sectionMap,
                            "std::map<mlir::Operation*, elf::writer::Symbol*>&":$symbolMap)
        >,
    ];
}

def ELF_SymbolObjectInterface : OpInterface<"ElfSymbolObjectInterface"> {
    let description = "Interface for ELF symbol objects";

    let cppNamespace = "vpux::ELF";

    let methods = [
        InterfaceMethod<
            "Serialize an object as an elf symbol",
            "void",
            "serialize", (ins "elf::writer::Symbol*":$symbol,
                            "vpux::OpOrderedMap<elf::writer::Section*>&":$sectionMap)
        >,
    ];
}

def ELF_RelocationObjectInterface : OpInterface<"ElfRelocationObjectInterface"> {
    let description = "Interface for ELF relocation objects";

    let cppNamespace = "vpux::ELF";

    let methods = [
        InterfaceMethod<
            "Serialize an object as an elf relocation object",
            "void",
            "serialize", (ins "elf::writer::Relocation*":$relocation,
                                "std::map<mlir::Operation*, elf::writer::Symbol*>&":$symbolMap)
        >,
    ];
}

//
// ELFObjectInterface
//

def ELF_OpInterface : OpInterface<"ElfObjectInterface"> {
    let description = "Interface for ELF object";

    let cppNamespace = "vpux::ELF";

    let methods = [
        InterfaceMethod<
            "Serialize the Task to BLOB format",
            "void",
            "serialize", (ins "elf::Writer&":$writer)
        >,
    ];
}

//
// BinaryOpInterface
//

def ELF_BinaryOpInterface : OpInterface<"BinaryOpInterface"> {
    let description = "Interface for ELF binary serialization of operations";
    let cppNamespace = "vpux::ELF";

    let methods = [
        InterfaceMethod<
            "Serialize the Task to BLOB format",
            "void",
            "serialize", (ins "std::vector<char>&":$buffer)
        >,
    ];
}


#endif
