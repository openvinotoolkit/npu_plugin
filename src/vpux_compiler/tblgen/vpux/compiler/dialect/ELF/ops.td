//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_ELF_OPS
#define VPUX_COMPILER_DIALECT_ELF_OPS

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/dialect/ELF/attributes.td"
include "vpux/compiler/dialect/ELF/dialect.td"
include "vpux/compiler/dialect/ELF/types.td"
include "vpux/compiler/dialect/VPUIPRegMapped/dialect.td"
include "vpux/compiler/dialect/VPUIPRegMapped/types.td"
include "mlir/IR/RegionKindInterface.td"
include "vpux/compiler/dialect/VPUIP/dialect.td"
include "vpux/compiler/dialect/VPURT/types.td"
include "vpux/compiler/dialect/ELF/ops_interfaces.td"


include "mlir/Interfaces/SideEffectInterfaces.td"


//
// Base class for the ELF dialect operations.
//

class ELF_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<
        ELF_Dialect,
        mnemonic,
        traits
    >;

//
// CreateSectionOp
//

def CreateSectionOp :
        ELF_Op<"CreateSection",
            [
                NoRegionArguments,
                DeclareOpInterfaceMethods<ELF_SectionInterface>
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Create ELF Section";

    let arguments = (ins
        StrAttr:$secName,
        ELF_SectionTypeAttr:$secType,
        ELF_SectionFlagsAttr:$secFlags,
        IntAttr:$secInfo,
        IntAttr:$secAddrAlign
        );

    let results = (outs ELF_Section:$section);

    let regions = (region
        SizedRegion<1>:$aRegion
    );

    let assemblyFormat = [{
        `secType` `(` $secType `)`
        `secFlags` `(` $secFlags `)`
        attr-dict
        `->` type(results)
        $aRegion
    }];
}

//
// CreateLogicalSectionOp :
//

def CreateLogicalSectionOp :
        ELF_Op<"CreateLogicalSection",
            [
                NoRegionArguments,
                DeclareOpInterfaceMethods<ELF_SectionInterface>
            ]
            # GraphRegionNoTerminator.traits
        > {

    let summary = "Create an ELF Logical Section, with no actual binary content in the ELF file ";

    let arguments = (ins
        StrAttr:$secName,
        ELF_SectionTypeAttr:$secType,
        ELF_SectionFlagsAttr:$secFlags,
        IntAttr:$secInfo,
        IntAttr:$secAddrAlign
        );

    let results = (outs ELF_Section:$section);

    let regions = (region
        SizedRegion<1>:$declaredOps
    );

    let assemblyFormat = [{
        `secType` `(` $secType `)`
        `secFlags` `(` $secFlags `)`
        attr-dict
        `->` type(results)
        $declaredOps
    }];
}

//
// CreateMetadataSectionOp
//

def CreateMetadataSectionOp :
        ELF_Op<"CreateMetadataSection",
            [
                NoRegionArguments,
                DeclareOpInterfaceMethods<ELF_SectionInterface>
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Create ELF Metadata Section";

    let arguments = (ins
        StrAttr:$secName,
        ELF_SectionFlagsAttr:$secFlags,
        IntAttr:$secInfo,
        IntAttr:$secAddrAlign
        );

    let results = (outs ELF_Section:$section);

    let regions = (region
        SizedRegion<1>:$aRegion
    );

    let extraClassDeclaration = [{
        void serialize(elf::Writer& writer, vpux::DenseMap<mlir::Operation*, elf::writer::Section*>& sectionMap, vpux::DenseMap<mlir::Operation*, elf::writer::Symbol*>& symbolMap, ELF::NetworkMetadata& metadata);
    }];

    let assemblyFormat = [{
        `secFlags` `(` $secFlags `)`
        attr-dict
        `->` type(results)
        $aRegion
    }];
}

//
// CreateSymbolTableSectionOp
//

def CreateSymbolTableSectionOp :
        ELF_Op<"CreateSymbolTableSection",
            [
                NoRegionArguments,
                DeclareOpInterfaceMethods<ELF_SectionInterface>
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Create ELF Symbol Table Section";

    let arguments = (ins
                      StrAttr:$secName,
                      ELF_SectionFlagsAttr:$secFlags,
                      OptionalAttr<UnitAttr>:$isBuiltin
                    );

    let results = (outs ELF_Section:$section);

    let regions = (region
        SizedRegion<1>:$aRegion
    );

    let assemblyFormat = [{
        `secName` `(` $secName `)`
        `secFlags` `(` $secFlags `)`
        attr-dict
        `->` type(results)
        $aRegion
    }];
}

//
// CreateRelocationSectionOp
//

def CreateRelocationSectionOp :
        ELF_Op<"CreateRelocationSection",
            [
                NoRegionArguments,
                DeclareOpInterfaceMethods<ELF_SectionInterface>
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Create ELF Relocation Section";

    let arguments = (ins
                      StrAttr:$secName,
                      ELF_Section:$sourceSymbolTableSection,
                      ELF_Section:$targetSection,
                      ELF_SectionFlagsAttr:$secFlags
                    );

    let results = (outs ELF_Section:$section);

    let regions = (region
        SizedRegion<1>:$aRegion
    );

    let assemblyFormat = [{
        `secName` `(` $secName `)`
        `sourceSymbolTableSection` `(` $sourceSymbolTableSection `)`
        `targetSection` `(` $targetSection `)`
        `secFlags` `(` $secFlags `)`
        attr-dict
        `->` type(results)
        $aRegion
    }];
}


//
// SymbolOp
//

def SymbolOp :
        ELF_Op<"Symbol",
            [
                DeclareOpInterfaceMethods<ELF_SymbolObjectInterface>
            ]
        > {
    let summary = "Create ELF Symbol Table Section";

    let arguments = (ins
          AnyType:$inputArg,
          OptionalAttr<UnitAttr>:$isBuiltin,
          OptionalAttr<StrAttr>:$name,
          OptionalAttr<ELF_SymbolTypeAttr>:$type,
          OptionalAttr<UI64Attr>:$size,
          OptionalAttr<UI64Attr>:$value
        );

    let results = (outs ELF_Symbol:$symbol);

    let assemblyFormat = [{
        $inputArg
        (`name` `(` $name^ `)`)?
        (`type` `(` $type^ `)`)?
        (`size` `(` $size^ `)`)?
        attr-dict
        `:` type($inputArg)
    }];
}

//
// RelocOp
//

def RelocOp :
    ELF_Op<"Reloc",
        [
            DeclareOpInterfaceMethods<ELF_RelocationObjectInterface>
        ]
    > {
    let summary = "Reloc Op";

    let arguments = (ins
          AnyType:$baseOp,
          AnyType:$offsetOf,
          ELF_RelocationTypeAttr:$relocationType,
          ELF_Symbol:$sourceSymbol,
          IntAttr:$addend
        );

    let results = (outs);

    let assemblyFormat = [{
        `baseOp` `(` $baseOp `:` type($baseOp) `)`
        `offsetOf` `(` $offsetOf `:` type($offsetOf) `)`
        $relocationType $sourceSymbol $addend attr-dict}];
}

def RelocImmOffsetOp :
    ELF_Op<"RelocImmOffset",
        [
            DeclareOpInterfaceMethods<ELF_RelocationObjectInterface>
        ]
    > {
    let summary = "Immediate Value Reloc Op";

    let arguments = (ins
          Optional<VPUIPRegMapped_IndexType>:$baseOp,
          IntAttr:$offset,
          ELF_RelocationTypeAttr:$relocationType,
          ELF_Symbol:$sourceSymbol,
          IntAttr:$addend
        );

    let results = (outs);

    let assemblyFormat = [{
        (`baseOp` `(` $baseOp^ `:` type($baseOp) `)`)?
        `offset` `(` $offset `)`
        $relocationType $sourceSymbol $addend attr-dict}];

    let extraClassDeclaration = [{
        uint32_t getKernelEntry();
        void serialize(elf::writer::Relocation* relocation, vpux::DenseMap<mlir::Operation*, elf::writer::Symbol*>& symbolMap, mlir::Operation* targetSection);
    }];


    }

//
// PutOpInSectionOp
//

def PutOpInSectionOp :
        ELF_Op<"PutOpInSection",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface>
            ]
        > {
    let summary = "Put the Argument Op in the ELF Section";

    let arguments = (ins
          AnyType:$inputArg
        );

    let results = (outs);

    let assemblyFormat = "$inputArg attr-dict `:` type($inputArg)";
}

#endif
