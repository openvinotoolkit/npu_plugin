//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_ELF_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_ELF_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// ELFSectionInterface
//

def ELF_SectionInterface : OpInterface<"ElfSectionInterface"> {
    let description = "Interface for the ELF section";

    let cppNamespace = "vpux::ELF";

    let methods = [
        InterfaceMethod<
            "Serialize the Section into an ELF entity",
            "void",
            "serialize", (ins "elf::Writer&":$writer,
                              "vpux::DenseMap<mlir::Operation*, elf::writer::Section*>&":$sectionMap,
                              "vpux::DenseMap<mlir::Operation*, elf::writer::Symbol*>&":$symbolMap)
        >,
        InterfaceMethod<
            "Get the block of the region",
            "mlir::Block*",
            "getBlock", (ins), [{
                return vpux::ELF::getSectionBlock($_op);
            }]
        >,
    ];
}

def ELF_SymbolObjectInterface : OpInterface<"ElfSymbolObjectInterface"> {
    let description = "Interface for the ELF symbol objects";

    let cppNamespace = "vpux::ELF";

    let methods = [
        InterfaceMethod<
            "Serialize an object as an ELF symbol",
            "void",
            "serialize", (ins "elf::writer::Symbol*":$symbol,
                              "vpux::DenseMap<mlir::Operation*, elf::writer::Section*>&":$sectionMap)
        >,
    ];
}

def ELF_RelocationObjectInterface : OpInterface<"ElfRelocationObjectInterface"> {
    let description = "Interface for the ELF relocation objects";

    let cppNamespace = "vpux::ELF";

    let methods = [
        InterfaceMethod<
            "Serialize an object as an ELF relocation object",
            "void",
            "serialize", (ins "elf::writer::Relocation*":$relocation,
                              "vpux::DenseMap<mlir::Operation*, elf::writer::Symbol*>&":$symbolMap)
        >,
    ];
}

//
// OpInterface
//

def ELF_OpInterface : OpInterface<"ElfObjectInterface"> {
    let description = "Interface for the ELF object";

    let cppNamespace = "vpux::ELF";

    let methods = [
        InterfaceMethod<
            "Serialize the Op to the BLOB format",
            "void",
            "serialize", (ins "elf::Writer&":$writer)
        >,
    ];
}

//
// BinaryOpInterface
//

def ELF_BinaryOpInterface : OpInterface<"BinaryOpInterface"> {
    let description = "Interface for the ELF binary serialization of operations";
    let cppNamespace = "vpux::ELF";

    let methods = [
        InterfaceMethod<
            "Serialize the Op to the BLOB format",
            "void",
            "serialize", (ins "elf::writer::BinaryDataSection<uint8_t>&":$binDataSection)
        >,

        InterfaceMethod<
            "Get the size in bytes of the serialized form of this object",
            "size_t",
            "getBinarySize", (ins)
        >,
    ];
}


def ELF_GetOffsetOfOpInterface : OpInterface<"GetOffsetOfOpInterface"> {
    let description = "Interface for the offset computation of different fields of inside serializable operations";
    let cppNamespace = "vpux::ELF";

    let methods = [
        InterfaceMethod<
            "Returns the offset of a field (specified by the 'val' parameter) inside a serializable op",
            "mlir::FailureOr<uint64_t>",
            "getOffsetOfWithinOperation", (ins "mlir::Value":$val)
        >,
    ];
}



#endif
