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
        ELF_SectionTypeAttr2:$secType,
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
        `secFlags` `(` $secFlags `)`
        attr-dict
        `->` type(results)
        $declaredOps
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
                      ELF_SectionFlagsAttr:$secFlags
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
          IntAttr:$offsetTargetField,
          ELF_RelocationTypeAttr:$relocationType,
          ELF_Symbol:$sourceSymbol,
          IntAttr:$addend
        );

    let results = (outs);

    let assemblyFormat = "$offsetTargetField $relocationType $sourceSymbol $addend attr-dict";
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
