#ifndef ELF_COMPILER_DIALECT_ELF
#define ELF_COMPILER_DIALECT_ELF

include "mlir/IR/OpBase.td"
//include "mlir/Interfaces/SideEffectInterfaces.td"


// Alex: This def is not included in Ops.h.inc, but here in this folder in
//   Dialect.h and in ../mlir/Dialect.cpp (the implementation of the ToyDialect
//   constructor).
// Provide a definition of the 'ELF' dialect in the ODS framework so that we
// can define our operations.
def ELF_Dialect : Dialect {
  let name = "ELF";

  //let cppNamespace = "::mlir::elf";
  let cppNamespace = "::vpux::ELF";

  let dependentDialects = [
    /*
    "vpux::Const::ConstDialect",
    "vpux::IE::IEDialect",
    "vpux::IERT::IERTDialect",
    "mlir::StandardOpsDialect",
    "mlir::quant::QuantizationDialect"
    */
    "vpux::IERT::IERTDialect",
    "vpux::VPUIPRegMapped::VPUIPRegMappedDialect"
  ];
}

#endif
