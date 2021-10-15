#ifndef ELF_COMPILER_DIALECT_ELF_OPS
#define ELF_COMPILER_DIALECT_ELF_OPS

include "vpux/compiler/core/attributes.td" // Alex: for IntAttr
include "vpux/compiler/dialect/ELF/attributes.td" // 2021_10_14
include "vpux/compiler/dialect/ELF/dialect.td"
include "vpux/compiler/dialect/ELF/types.td" // 2021_10_01
include "mlir/IR/RegionKindInterface.td" // 2021_10_05
include "vpux/compiler/dialect/VPUIP/dialect.td" // 2021_10_06
include "vpux/compiler/dialect/VPUIP/types.td" // 2021_10_06
/*
include "vpux/compiler/dialect/VPUIP/attributes.td"
include "vpux/compiler/dialect/VPUIP/dialect.td"
include "vpux/compiler/dialect/VPUIP/ops_interfaces.td"
include "vpux/compiler/dialect/VPUIP/types.td"
*/
include "vpux/compiler/dialect/VPUIPRegMapped/types.td" // 2021_10_11

include "mlir/Interfaces/SideEffectInterfaces.td"







// Alex: This def is sort of included in Ops.h.inc
// Base class for toy dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class ELF_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<ELF_Dialect, mnemonic, traits>;



// Alex: CreateSectionOp is inspired from TransposeOp.
def CreateSectionOp : ELF_Op<"CreateSection",
            [
                //ViewLikeOpInterface, // From https://reviews.llvm.org/D78645: should add: include "mlir/Interfaces/ViewLikeInterface.td"
                NoRegionArguments
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Create Section";

    let arguments = (ins
          //StrAttr:$id,
          //IntAttr:$secType,
          // Gives <<error: ‘SectionTypeAttr’ in namespace ‘vpux::ELF’ does not name a type>>: ELF_SectionType:$secType,
          // OK: AnyAttr:$secType, // Inspired from "def IERT_ExecutorResourceOp" from IERT/ops.td
          // Gives following error while parsing: <<'ELF.CreateSection' op attribute 'secType' failed to satisfy constraint>>:
          //<<error: 'ELF.CreateSection' op attribute 'secType' failed to satisfy constraint>>: ELF_SectionTypeAttr_BitEnum:$secType, // Inspired from "def VPUIP_GraphOp" from VPUIP/ops.td
          ELF_SectionTypeAttr:$secType, // Inspired partly from "def VPUIP_GraphOp" from VPUIP/ops.td

          //IntAttr:$secFlags
          ELF_SectionFlagsAttr:$secFlags,

          IntAttr:$secInfo,

          IntAttr:$secAddrAlign

          //entry_size
        );

    let results = (outs ELF_Section:$section);
    // This compiles: let results = (outs Index:$section); // Got inspired from OpBase.td: def Index : Type<CPred<"$_self.isa<::mlir::IndexType>()">, "index"...
    // <<error: Variable not defined: 'index'>>: let results = (outs index:$section);
    // <<error: Variable not defined: 'IndexType'>>: let results = (outs IndexType:$section);
    // Did not work well: let results = (outs AnyMemRef:$section);
    //let results = (outs VPUIP_Barrier:$section);

    // Enable registering canonicalization patterns with this operation.
    //let hasCanonicalizer = 1;

    /*
    // Inspired also from the definition of VPUIP_NCEClusterTaskOp
    let regions = (region
        SizedRegion<1>:$nndmaOps,
        SizedRegion<1>:$configureBarrierOps
    );
    */
    /*
    */
    let regions = (region
        SizedRegion<1>:$aRegion
    );
    /*
    */
    // Inspired also from the definition of VPUIP_NCEClusterTaskOp
    let assemblyFormat = [{
        attr-dict
        `->` type(results)
        $aRegion
    }];
       /*
        `secType` `(` $secType `)`
        `secFlags` `(` $secFlags `)`
       */
      /*
        // Good, but canNOT reference sub-region variables from another sub-region
        $aRegion
      */
      /*
       Gives:
        <<error: 'ELF.CreateSection' op region #0 ('nndmaOps') failed to verify constraint: region with 1 blocks>>
         due to the CreateSectionOp::verify() inside ops.cpp.inc:
        (`configureBarrierOps` `:` $configureBarrierOps^)?
        (`nndmaOps` `:` $nndmaOps^)?
      */
      /*
       OK (works), but NOT really good:
        `configureBarrierOps` `:` $configureBarrierOps
        `nndmaOps` `:` $nndmaOps
      */
      // <<error: optional group specified no anchor element>>: (`nndmaOps` `:` $nndmaOps)?
      // Good: `<` $type `,` $flags `>`
      // error: expected valid literal: `<` `type=` $type `,` `flags=` $flags `>`
      // error: expected valid literal: `<type=` $type `,flags=` $flags `>`
      // error: expected valid literal: (`<type=` $type `,flags=` $flags `>`)
      // error: expected valid literal: (`type=` $type `,flags=` $flags)
      /*
      error: expected '?' after optional group
          (`flags` `(` $flags `)`)
       (`type` `(` $type `)`)
       (`flags` `(` $flags `)`)
       */
      /*
      error: expected '?' after optional group
            `->` type(results)
       (`flags` `(` $flags `)`
        `type` `(` $type `)`)
      */
      //(`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
      //`->` type(results)


    /*
    // 2021_08_20
    // Specify a parser and printer method.
    let parser = [{ return ::parseConstantOp(parser, result); }];
    let printer = [{ return ::print(p, *this); }];
    */

    /*
    // Allow building a CreateSectionOp from the input operand(s).
    let builders = [
        //OpBuilderDAG<(ins "Value":$input)>
        //OpBuilder<(ins "Value":$input)>
        //OpBuilder<(ins "::mlir::Value":$input)>
        // GOOD: OpBuilder<(ins)>
        OpBuilder<(ins "int64_t":$secType,
                       "int64_t":$secFlags)>
    ];
    */

    /*
    // Invoke a static verify method to verify this absolute operation.
    let verifier = [{
        //return ::verify(*this);
        // TODO (defined in many files in kmb-plugin-elf/src/vpux_compiler/src/dialect/VPUIP/ops, and nce_invariant.cpp, and declared in kmb-plugin-elf/src/vpux_compiler/include/vpux/compiler/dialect/VPUIP/ops.hpp): return vpux::ELF::verifyOp(*this);
    }];
    */
}



def PutAnyOpInSectionOp : ELF_Op<"PutAnyOpInSection"> {
    let summary = "Put Any Op in ELF Section";

    let arguments = (ins
          //VPUIPRegMapped_Barrier:$barrier
          //AnyType:$result // Inspired from mlir/Dialect/StandardOps/IR/Ops.td, "def AtomicYieldOp"
          AnyType:$inputArg // Inspired from mlir/Dialect/StandardOps/IR/Ops.td, "def AtomicYieldOp"
        );

    let results = (outs);

    // Enable registering canonicalization patterns with this operation.
    //let hasCanonicalizer = 1;
    /*
    let assemblyFormat = [{
        attr-dict
        $barrier
    }];
    */



  let assemblyFormat = "$inputArg attr-dict `:` type($inputArg)"; // Inspired from mlir/Dialect/StandardOps/IR/Ops.td, "def AtomicYieldOp"
  //let assemblyFormat = "$result attr-dict `:` type($result)"; // Inspired from mlir/Dialect/StandardOps/IR/Ops.td, "def AtomicYieldOp"
  // Gives <<error: type of operand #0, named 'result', is not buildable and a buildable type cannot be inferred>>: let assemblyFormat = "$result attr-dict";

    /*
    // 2021_08_20
    // Specify a parser and printer method.
    let parser = [{ return ::parseConstantOp(parser, result); }];
    let printer = [{ return ::print(p, *this); }];
    */

    /*
    // Allow building a CreateSectionOp from the input operand(s).
    let builders = [
        //OpBuilderDAG<(ins "Value":$input)>
        //OpBuilder<(ins "Value":$input)>
        //OpBuilder<(ins "::mlir::Value":$input)>
        // GOOD: OpBuilder<(ins)>
        OpBuilder<(ins "int64_t":$secType,
                       "int64_t":$secFlags)>
    ];
    */

    /*
    // Invoke a static verify method to verify this absolute operation.
    let verifier = [{
        //return ::verify(*this);
        // TODO (defined in many files in kmb-plugin-elf/src/vpux_compiler/src/dialect/VPUIP/ops, and nce_invariant.cpp, and declared in kmb-plugin-elf/src/vpux_compiler/include/vpux/compiler/dialect/VPUIP/ops.hpp): return vpux::ELF::verifyOp(*this);
    }];
    */
}





def CreateSectionSymTableOp : ELF_Op<"CreateSectionSymTable"> {
    let summary = "Create Section SymTable";

    let arguments = (ins);
    let results = (outs);

    // Enable registering canonicalization patterns with this operation.
    //let hasCanonicalizer = 1;

    /*
    let assemblyFormat = [{
    }];
    */

    /*
    // 2021_08_20
    // Specify a parser and printer method.
    let parser = [{ return ::parseConstantOp(parser, result); }];
    let printer = [{ return ::print(p, *this); }];
    */

    /*
    // Allow building a CreateSectionSymTableOp from the input operand.
    let builders = [
        //OpBuilderDAG<(ins "Value":$input)>
        //OpBuilder<(ins "::mlir::Value":$input)>
        // Gives: <<error: ‘static void vpux::ELF::CreateSectionSymTableOp::build(mlir::OpBuilder&, mlir::OperationState&)’ cannot be overloaded with ‘static void vpux::ELF::CreateSectionSymTableOp::build(mlir::OpBuilder&, mlir::OperationState&)’>>: OpBuilder<(ins)>
    ];
    */

    /*
    // Invoke a static verify method to verify this absolute operation.
    let verifier = [{ return ::verify(*this); }];
    */
}


def CreateSectionRelocTableOp : ELF_Op<"CreateSectionRelocTable"> {
    let summary = "Create Section RelocaTable";

    let arguments = (ins);
    let results = (outs);

    // Enable registering canonicalization patterns with this operation.
    //let hasCanonicalizer = 1;

    /*
    let assemblyFormat = [{
    }];
    */

    /*
    // 2021_08_20
    // Specify a parser and printer method.
    let parser = [{ return ::parseConstantOp(parser, result); }];
    let printer = [{ return ::print(p, *this); }];
    */

    /*
    // Allow building a CreateSectionRelocTableOp from the input operand(s).
    let builders = [
        //OpBuilderDAG<(ins "Value":$input)>
        //OpBuilder<(ins "Value":$input)>
        //OpBuilder<(ins "::mlir::Value":$input)>
        OpBuilder<(ins)>
    ];
    */

    /*
    // Invoke a static verify method to verify this absolute operation.
    let verifier = [{ return ::verify(*this); }];
    */
}


#endif
