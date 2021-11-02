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

#ifndef VPUX_COMPILER_DIALECT_ELF_OPS
#define VPUX_COMPILER_DIALECT_ELF_OPS

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/dialect/ELF/attributes.td"
include "vpux/compiler/dialect/ELF/dialect.td"
include "vpux/compiler/dialect/ELF/types.td"
include "mlir/IR/RegionKindInterface.td"
include "vpux/compiler/dialect/VPUIP/dialect.td"
include "vpux/compiler/dialect/VPUIP/types.td"
include "vpux/compiler/dialect/VPUIPRegMapped/types.td"

include "mlir/Interfaces/SideEffectInterfaces.td"


//
// Base class for the ELF dialect operations.
//

class ELF_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<ELF_Dialect, mnemonic, traits>;

//
// CreateSectionOp
//

def CreateSectionOp : ELF_Op<"CreateSection",
            [
                //ViewLikeOpInterface, // From https://reviews.llvm.org/D78645: should add: include "mlir/Interfaces/ViewLikeInterface.td"
                NoRegionArguments
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Create ELF Section";

    let arguments = (ins
        StrAttr:$secName,
        ELF_SectionTypeAttr:$secType, // Inspired partly from "def VPUIP_GraphOp" from VPUIP/ops.td
        ELF_SectionFlagsAttr:$secFlags,
        IntAttr:$secInfo,
        IntAttr:$secAddrAlign
        );

    let results = (outs ELF_Section:$section);

    // Inspired also from the definition of VPUIP_NCEClusterTaskOp
    let regions = (region
        SizedRegion<1>:$aRegion
    );

    // Inspired also from the definition of VPUIP_NCEClusterTaskOp
    let assemblyFormat = [{
        `secFlags` `(` $secFlags `)`
        attr-dict
        `->` type(results)
        $aRegion
    }];
    //    $secFlags
}

//
// PutAnyOpInSectionOp
//

def PutAnyOpInSectionOp : ELF_Op<"PutAnyOpInSection"> {
    let summary = "Put Any Op in ELF Section";

    let arguments = (ins
          AnyType:$inputArg // Inspired from mlir/Dialect/StandardOps/IR/Ops.td, "def AtomicYieldOp"
        );

    let results = (outs);

    let assemblyFormat = "$inputArg attr-dict `:` type($inputArg)"; // Inspired from mlir/Dialect/StandardOps/IR/Ops.td, "def AtomicYieldOp"
    // Gives <<error: type of operand #0, named 'result', is not buildable and a buildable type cannot be inferred>>: let assemblyFormat = "$result attr-dict";
}


//
// SymbolOp
//

def SymbolOp : ELF_Op<"Symbol"> {
    let summary = "Create ELF Symbol Table Section";

    let arguments = (ins
          AnyType:$inputArg // Inspired from mlir/Dialect/StandardOps/IR/Ops.td, "def AtomicYieldOp"
        );

    let results = (outs ELF_Symbol:$symbol);

    // Inspired also from the definition of VPUIP_NCEClusterTaskOp
    let assemblyFormat = [{
        $inputArg attr-dict `:` type($inputArg)
    }];
    /*
       Error: <<suggest adding a type constraint to the operation or adding a 'type($inputArg)' directive to the custom assembly format>>
        $inputArg
        attr-dict
    */
     //   $inputArg attr-dict `:` type($inputArg)
     //   `->` type(results)
}


//
// CreateSymbolTableSectionOp
//

def CreateSymbolTableSectionOp : ELF_Op<"CreateSymbolTableSection",
            [
                NoRegionArguments
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Create ELF Symbol Table Section";

    let arguments = (ins
                      StrAttr:$secName
                    );

    let results = (outs ELF_Section:$section);

    // Inspired also from the definition of VPUIP_NCEClusterTaskOp
    let regions = (region
        SizedRegion<1>:$aRegion
    );

    // Inspired also from the definition of VPUIP_NCEClusterTaskOp
    let assemblyFormat = [{
        attr-dict
        `->` type(results)
        $aRegion
    }];
      //  `->` type(results)
}


//
// CreateRelocationSectionOp
//

def CreateRelocationSectionOp : ELF_Op<"CreateRelocationSection",
            [
                NoRegionArguments
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Create ELF Relocation Section";

    let arguments = (ins
                      StrAttr:$secName,
                      //ELF_Symbol:$sourceSymbolTable,
                      ELF_Section:$sourceSymbolTableSection,
                      ELF_Section:$targetSection,
                      ELF_SectionFlagsAttr:$secFlags
                    );

    let results = (outs ELF_Section:$section);

    // Inspired also from the definition of VPUIP_NCEClusterTaskOp
    let regions = (region
        SizedRegion<1>:$aRegion
    );

    // Inspired also from the definition of VPUIP_NCEClusterTaskOp
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
// RelocOp
//

def RelocOp : ELF_Op<"Reloc"> {
    let summary = "Reloc Op";

    let arguments = (ins
          IntAttr:$offsetTargetField,
          //StrAttr:$relocationType,
          ELF_RelocationTypeAttr:$relocationType,
          ELF_Symbol:$sourceSymbol,
          IntAttr:$addend //$offsetSourceField
        );

    let results = (outs);

    let assemblyFormat = "$offsetTargetField $relocationType $sourceSymbol $addend attr-dict";
    //let assemblyFormat = "$inputArg attr-dict `:` type($inputArg)"; // Inspired from mlir/Dialect/StandardOps/IR/Ops.td, "def AtomicYieldOp"
}


#endif
