//
// Copyright 2021 Intel Corporation.
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

#ifndef VPUX_COMPILER_DIALECT_ELF
#define VPUX_COMPILER_DIALECT_ELF

include "mlir/IR/OpBase.td"

// Provide a definition of the ELF dialect in the ODS framework
def ELF_Dialect : Dialect {
  let name = "ELF";

  let cppNamespace = "::vpux::ELF";

  let dependentDialects = [
    "vpux::IERT::IERTDialect",
    "vpux::VPUIPRegMapped::VPUIPRegMappedDialect"
  ];
}

#endif
